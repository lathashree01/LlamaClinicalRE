import os
import logging

import pandas as pd
import wandb
import optuna
# from optuna.pruners import ThresholdPruner

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from transformers import LlamaForSequenceClassification,LlamaTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model,PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,precision_recall_fscore_support

# Initialize wandb
os.environ["WANDB_API_KEY"]=""
os.environ["WANDB_ENTITY"]="lathashree01"
os.environ["WANDB_PROJECT"]="hyperparam-pretrain-llama"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_MODE"]="dryrun"

# Configure the logging settings
logging.basicConfig(level=logging.INFO, filename='hyperparam_tuning_euadr.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda')

# hard coded params
data_path='/homes/l22/Documents/git_repos/thesis/datasets/'

PATH_TO_CONVERTED_WEIGHTS='meta-llama/Llama-2-7b-hf'
mycache_dir="/vol/bitbucket/l22/llama2"

PEFT_PATH="/vol/bitbucket/l22/llama2_pretrained/pt_lora_model"
# PEFT_PATH="/vol/bitbucket/l22/llama1_pretrained/pt_lora_model"

# Hardcoded hyperparameters
SEED = 42
num_epochs = 10
lr=2e-4
gradient_accumulation_steps=2
max_length=256
batch_size=2
# lora_rank=4
# lora_alpha=16
lora_trainable="q_proj,v_proj,k_proj,o_proj"
modules_to_save="embed_tokens,lm_head"
# lora_dropout=0.05
target_modules = lora_trainable.split(',')
mod_to_save = modules_to_save.split(',')
step = 0
best_val_f1 = 0
torch.manual_seed(SEED)
test_scores={}

full_data = None


def get_f1_score(out_logits, true_labels):
    """ Compute the f1 score given the predicted logits and true labels """
    pred_probs = F.softmax(out_logits, dim=-1)
    predicted_labels = torch.argmax(pred_probs.float(), dim=1)
    f1 = f1_score(true_labels.tolist(), predicted_labels.tolist(), average='micro')
    return f1


def comput_test_metrics(y_true, y_preds,benchmark):
    """ Compute accuracy, precision, recall and f1 score for the given predictions; create predictions_file"""

    macro_vals = precision_recall_fscore_support(y_true, y_preds, average='macro')
    micro_vals = precision_recall_fscore_support(y_true, y_preds, average='micro')
    weighted_vals = precision_recall_fscore_support(y_true, y_preds, average='weighted')
    class_vals = precision_recall_fscore_support(y_true, y_preds, average=None)

    print("| Precision | Recall | F1 Score | Support |")
    print("Test Scores (micro avg): {}".format(micro_vals))
    print("Test Scores (macro avg): {}".format(macro_vals))
    print("Test Scores (weighted avg): {}".format(weighted_vals))
    print("Test Scores (class avg): {}".format(class_vals))

    # with open(data_path + benchmark +"/test_results.txt", "w") as test_results:
    #     test_results.writelines(str(y_preds))
    #     test_results.write("\n")
    #     test_results.writelines(str(y_true))


def read_datafile(filepath):
    """ Read the data from the given filepath """
    df = pd.read_csv(filepath, header=None, delimiter='\t', names=['sentence', 'label'])
    return df


def load_train_data(filepath, tokenizer, maxlen=max_length, train_percentage=0.8, mode="train"):
    """
    Params:
        filepath: path of the dataset
        tokenizer: tokenizer to use
        max_length: maxlength of text (default=512)
        train_percentage: percentage of data to use for training (default=0.7)
        mode: "train" or "test" (default=train)
    """
    # load dataset through csv file
    print("Loading data from {}".format(filepath))
    df = pd.read_csv(filepath, header=None, delimiter='\t', names=['sentence', 'label'])

    sentences = df.sentence.values
    labels = df.label.values
    
    input_ids = []
    attention_masks = []
    
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=maxlen,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # convert lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_tensor = torch.tensor(labels)


    dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
    if mode == "train":
        train_size = int(train_percentage * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print("Train size: {}".format(train_size))
        print("Validation size: {}".format(val_size))
        logging.info("Train size: {}".format(train_size))
        logging.info("Validation size: {}".format(val_size))
        
        return (train_dataset, val_dataset)
    else:
        print("Test size: {}".format(len(dataset)))
        return dataset

def train_euadr_model(model, data, benchi,optimizer=Adam, b_size=batch_size, epochs=num_epochs, lr=lr, gradient_accumulation_steps=gradient_accumulation_steps,eps=1e-8):
    """ Train the model on the given dataset 
    Params:
        model: model to train
        data: dataset
        benchi: benchmark dataset name
        optimizer: optimizer to use = Adam (default)
        batch_size: batch size 
        epochs: number of epochs   
        lr: Learning Rate 
    """
    global step, best_val_f1
    train_ds = data[0]
    val_ds = data[1]
    avg_val_f1 = 0
    avg_train_loss=0
    train_dataloader = DataLoader(
        train_ds,
        sampler=RandomSampler(train_ds),
        batch_size=b_size
    )

    val_dataloader = DataLoader(
        val_ds,
        sampler=SequentialSampler(val_ds),
        batch_size=b_size
    )
    model.train()
    optim = optimizer(model.parameters(), lr, eps=eps)

    scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * epochs
        )
    wandb.init()
    logging.info("Starting training...")
    for e in range(epochs):
        train_loss = 0
        for b_idx, batch in  enumerate(tqdm(train_dataloader)):
            step += 1
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            output = model(
                b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels
            )

            loss = output['loss']
            train_loss += loss.item()
            loss.backward()
            if (b_idx + 1) % gradient_accumulation_steps == 0:
                optim.step()
                scheduler.step()
                optim.zero_grad()
        
            wandb.log({"train_loss/step": loss.item()/b_size}, step=step)

        avg_train_loss = (train_loss / len(train_dataloader))/epochs
        logging.info('Average training loss for epoch: {}'.format(avg_train_loss))
        print('Average training loss for epoch: {}'.format(avg_train_loss))

        # validation
        logging.info("evaluating model...")
        val_loss = 0
        val_f1 = 0

        model.eval()
        for _,batch in enumerate(tqdm(val_dataloader)):
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                output = model(
                    b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
            
            loss = output['loss']
            f1_val = get_f1_score(output.logits, b_labels)
            val_loss += loss.item()
            val_f1 += f1_val

            wandb.log({"val_loss/step": loss.item()/b_size}, step=step)
            wandb.log({"val_f1/step": val_f1/b_size}, step=step)
        
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_f1 = val_f1 / len(val_dataloader)


        print('average validation loss for epoch: {}'.format(avg_val_loss))
        print('average validation F1 for epoch: {}'.format(avg_val_f1))
        logging.info('average validation loss and F1 for epoch {}: {} and {}'.format(e,avg_val_loss, avg_val_f1))

    return avg_val_f1


def test_euadr_model(model, data,benchmark, b_size=batch_size):
    """ Test the model on the given dataset
    Params:
        model: model to test
        data: test dataset
        b_size: batch size = 2 (default)
    """
    test_ds = data

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=b_size
    )

    logging.info("Testing  model...")
    test_loss = 0
    test_f1 = 0

    model.eval()
    
    # Storing true and predicted labels for computing metrics
    y_true = test_ds[:][2].tolist()
    y_pred=[]

    for new_step,batch in enumerate(tqdm(test_loader)):
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(
                b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )
        
        loss = output['loss']

        # Convert logits to probabilities and compute accuracy
        pred_probs = F.softmax(output.logits, dim=-1)
        predicted_labels = torch.argmax(pred_probs.float(), dim=1)
        f1 = get_f1_score(output.logits, b_labels)
        y_pred.extend(predicted_labels.tolist())

        test_loss += loss.item()
        test_f1 += f1
        wandb.log({"test_loss/step": loss.item()/b_size}, step=new_step)
        wandb.log({"test_f1/step": test_f1/b_size}, step=new_step)
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_f1 = test_f1 / len(test_loader)

    print('average test loss for epoch: {}'.format(avg_test_loss))
    print('average test f1 for epoch: {}'.format(avg_test_f1))
    logging.info('average test loss  and f1: {} and {}'.format(avg_test_loss, avg_test_f1))

    print("Final Stats:")
    comput_test_metrics(y_true, y_pred,benchmark)


def train_objective_function(trial):
    """ Objective function for hyperparameter tuning """

    # Hyperparameters to optimise
    params = {
        "gradient_accumulation_steps": trial.suggest_categorical('grad_acc_steps',[4,6,8]),
        "lr": trial.suggest_categorical('lr', [1e-4, 2e-4, 2e-5,5e-5]),
        "optimizer_name": trial.suggest_categorical('optimizer_name',["AdamW", "Adam"]),
        "eps": trial.suggest_categorical('eps',[1e-8, 2e-7, 1e-7])
    }

    print("Starting trial on euadr data with params: {}".format(params))

    print("Loading LLAMA model...")
    llama2_model = LlamaForSequenceClassification.from_pretrained(
        PATH_TO_CONVERTED_WEIGHTS,
        # cache_dir=mycache_dir,
        num_labels=2,
        torch_dtype=getattr(torch, 'bfloat16'),
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(llama2_model, PEFT_PATH)
    for param in model.parameters():
        if param.dtype == torch.float32 or param.dtype == torch.float16 :
            param.data = param.data.to(torch.bfloat16)
            
    model.print_trainable_parameters()
    model.to(device)

    optim = getattr(
        torch.optim, params["optimizer_name"]
    )
    
    avg_val_f1 = train_euadr_model(model, full_data, "euadr", optimizer=optim, b_size=batch_size, epochs=num_epochs, lr=params["lr"], gradient_accumulation_steps=params["gradient_accumulation_steps"], eps=params["eps"])
    torch.cuda.empty_cache()
    return avg_val_f1


if __name__ == '__main__':
    
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    tokenizer = LlamaTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    full_data = load_train_data(data_path + 'euadr/final_data/train.tsv', tokenizer=tokenizer, maxlen=256, mode="train")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(train_objective_function, n_trials=15)
    print('Best hyperparameters: ', study.best_params)
    print('Best val F1: ', study.best_value)

    hist_fig = optuna.visualization.plot_optimization_history(study)
    param_imp_fig = optuna.visualization.plot_param_importances(study)
    parallel_info_fig = optuna.visualization.plot_parallel_coordinate(study)

    hist_fig.write_image('../../visualisation/hp_tuning/euadr_hist_img.pdf')
    param_imp_fig.write_image("../../visualisation/hp_tuning/euadr_param_imp.pdf")
    parallel_info_fig.write_image("../../visualisation/hp_tuning/euadr_parallel_info.pdf")