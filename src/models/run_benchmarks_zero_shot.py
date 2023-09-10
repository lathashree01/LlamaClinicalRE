"""
    Python script to run zero-shot on benchmark datasets: GAD, euadr, i2b2 2010
    
"""

import os
import argparse

import pandas as pd
import wandb
import logging
import pickle
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from transformers import LlamaForSequenceClassification,LlamaTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model,PeftModel, PeftConfig, TaskType
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

# Initialize wandb
os.environ["WANDB_API_KEY"]=""
os.environ["WANDB_ENTITY"]="lathashree01"
os.environ["WANDB_PROJECT"]="zeroshot-benchi-llama"
# os.environ["WANDB_MODE"]="dryrun"

# Configure the logging settings
logging.basicConfig(level=logging.INFO, filename='zeroshot-benchi-llama.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda')
loss_dict={}
train_loss_list=[]
val_loss_list=[]
val_f1_list=[]
test_loss_list=[]
test_f1_list=[]

# hard coded params
data_path='datasets/'
model_path='../models/final_benchi_peft_models/'

# PEFT_PATH=None

# Original LLAMA 1 
# PATH_TO_ORIGINAL_WEIGHTS="/rds/general/user/l22/home/llama-hf/"

# Original LLAMA 2
# PATH_TO_ORIGINAL_WEIGHTS='meta-llama/Llama-2-7b-hf'
# mycache_dir='/rds/general/user/l22/home/llama_2/'

# Pretrained LLAMA 1
# PEFT_PATH='/vol/bitbucket/l22/llama1_pretrained/pt_lora_model'

# Pretrained LLAMA 2
# PEFT_PATH='/vol/bitbucket/l22/llama2_pretrained/pt_lora_model'


SEED = 42
num_epochs = 1
lr=2e-5
gradient_accumulation_steps=6
max_length=256
batch_size=2
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05
target_modules = lora_trainable.split(',')
mod_to_save = modules_to_save.split(',')
step = 0
torch.manual_seed(SEED)


def get_f1_score(out_logits, true_labels):
    """ Compute the f1 score given the predicted logits and true labels """
    pred_probs = F.softmax(out_logits, dim=-1)
    predicted_labels = torch.argmax(pred_probs.float(), dim=1)
    f1 = f1_score(true_labels.tolist(), predicted_labels.tolist(), average='micro')
    return f1


def comput_test_metrics(y_true, y_preds,benchmark):
    """ Compute precision, recall and f1 score for the given predictions; create predictions_file"""

    macro_vals = precision_recall_fscore_support(y_true, y_preds, average='macro')
    micro_vals = precision_recall_fscore_support(y_true, y_preds, average='micro')
    weighted_vals = precision_recall_fscore_support(y_true, y_preds, average='weighted')
    class_vals = precision_recall_fscore_support(y_true, y_preds, average=None)

    print("y True: {}".format(y_true))
    print("y Preds: {}".format(y_preds))

    print("| Precision | Recall | F1 Score | Support |")
    print("Test Scores (micro avg): {}".format(micro_vals))
    print("Test Scores (macro avg): {}".format(macro_vals))
    print("Test Scores (weighted avg): {}".format(weighted_vals))
    print("Test Scores (class avg): {}".format(class_vals))

    with open(data_path + benchmark +"/final_test_results.txt", "w") as test_results:
        test_results.writelines(str(y_preds))
        test_results.write("\n")
        test_results.writelines(str(y_true))


def load_data(filepath, tokenizer, train_percentage=0.8, mode="train"):
    """
    Args:
        filepath: path of the dataset
        tokenizer: tokenizer to use
        train_percentage: percentage of data to use for training (default=0.8)
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
            max_length=max_length,
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
#     print("Size of dataset : {}".format(len(dataset)))

    if mode == "train":
        train_size = int(train_percentage * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print("Train size: {}".format(train_size))
        print("Validation size: {}".format(val_size))
        logging.info("Train size: {}".format(train_size))
        logging.info("Validation size: {}".format(val_size))
        
        return (train_dataset, val_dataset)
    elif mode == "test":
        print("Test size: {}".format(len(dataset)))
        return dataset

def test_model(model, data,benchmark, b_size=batch_size):
    """ Test the model on the given dataset
    Params:
        model: model to test
        data: test dataset
        b_size: batch size = 2 (default)
    """
    test_dataset = data

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=b_size
    )

    logging.info("Testing  model...")
    test_loss = 0
    test_f1 = 0

    model.eval()
    
    # Storing true and predicted labels for computing metrics
    y_true = []
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
        
        y_true.extend(batch[2].tolist())
        loss = output['loss']

        test_loss_list.append(loss.item())
        
        # Convert logits to probabilities and compute f1
        pred_probs = F.softmax(output.logits, dim=-1)
        predicted_labels = torch.argmax(pred_probs.float(), dim=1)
        f1 = get_f1_score(output.logits, b_labels)
        test_f1_list.append(f1)
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

    print("Saving test loss dict")

    with open(benchmark+'_test_loss_dict.pickle', 'wb') as handle:
        loss_dict["test_loss"] = test_loss_list
        loss_dict["test_f1"] = test_f1_list
        pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        loss_dict.clear()
        test_loss_list.clear()
        test_f1_list.clear()

    print("Final Stats")
    comput_test_metrics(y_true, y_pred,benchmark)


def run_main(args):
    """
    Function that runs supervised fine-tuning on all benchmark datasets
    
    Args:
        args: argument parser object with 4 params - model_type, PATH_TO_ORIGINAL_WEIGHTS, PEFT_PATH, mycache_dir
    """
    print("Running for model : ", args.model_type)
    
    # Loading LLAMA tokeniser
    logging.info("Loading LLAMA tokeniser...")
    if args.model_type == "llama1" or args.model_type == "llama1_pre" :
        tokenizer = LlamaTokenizer.from_pretrained(args.PATH_TO_ORIGINAL_WEIGHTS,use_fast=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(args.PATH_TO_ORIGINAL_WEIGHTS,cache_dir=args.mycache_dir, use_fast=True)
        
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logging.info("Tokenizer loaded")

    benchmark = ["euadr", "GAD"]

    # ****************************** GAD. euadr ****************************** #

    for benchi in benchmark:
        logging.info("Loading LLAMA model for benchmark{}...from path {}".format(benchi,args.PATH_TO_ORIGINAL_WEIGHTS))
        print("PEFT path {}".format(args.PEFT_PATH))
        print("Loading LLAMA model for benchmark - {}...from path - {}".format(benchi,args.PATH_TO_ORIGINAL_WEIGHTS))
        if args.model_type == "llama1" or args.model_type == "llama1_pre":
            llama_model = LlamaForSequenceClassification.from_pretrained(
                args.PATH_TO_ORIGINAL_WEIGHTS,
                num_labels=2,
                torch_dtype=getattr(torch, 'bfloat16'),
                low_cpu_mem_usage=True,  
            )
        else:
            llama_model = LlamaForSequenceClassification.from_pretrained(
                args.PATH_TO_ORIGINAL_WEIGHTS,
                cache_dir=args.mycache_dir,
                num_labels=2,
                torch_dtype=getattr(torch, 'bfloat16'),
                low_cpu_mem_usage=True,  
            )
        
        # new peft model config
        new_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=mod_to_save)

        print(new_peft_config)
        
        # Load pretrained model of original llama models
        if args.model_type =="llama1_pre" or args.model_type == "llama2_pre": 
            model = PeftModel.from_pretrained(llama_model, args.PEFT_PATH, is_trainable=False)
            model = get_peft_model(model, peft_config=new_peft_config)
        else:
            model = get_peft_model(llama_model, peft_config=new_peft_config)
            
        model.print_trainable_parameters()

        # Convert model parameters to bfloat16
        for param in model.parameters():
            if param.dtype == torch.float32 or param.dtype == torch.float16 :
                param.data = param.data.to(torch.bfloat16)

        model.to(device)
        step = 0
        
        # Start wandb session and training
        wandb.init()
        # print("Starting Training for {}".format(benchi))
        # data = load_data(data_path + benchi +'/final_data/train.tsv', tokenizer=tokenizer, mode="train")
        # model = train_model(model, data, benchi)
        
        # Starting testing
        print("Starting Testing")
        logging.info("Starting Testing")
        data = load_data(data_path + benchi + '/final_data/test.tsv', tokenizer=tokenizer, mode="test")
        model = test_model(model, data, benchi)
        print("Completed Testing for {}".format(benchi))
        wandb.finish()

    # ****************************** i2b2 2010 ****************************** #

    print("Training on i2b2 data")
    logging.info("Training on i2b2 data")
    
    logging.info("Loading LLAMA model...")
    print("Loading LLAMA model...")
    
    if args.model_type == "llama1" or args.model_type == "llama1_pre":
        llama_model = LlamaForSequenceClassification.from_pretrained(
                args.PATH_TO_ORIGINAL_WEIGHTS,
                num_labels=8,
                torch_dtype=getattr(torch, 'bfloat16'),
                low_cpu_mem_usage=True,  
        )
    else:
        llama_model = LlamaForSequenceClassification.from_pretrained(
                args.PATH_TO_ORIGINAL_WEIGHTS,
                cache_dir=args.mycache_dir,
                num_labels=8,
                torch_dtype=getattr(torch, 'bfloat16'),
                low_cpu_mem_usage=True,  
        ) 
        
    # new peft model config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
        inference_mode=False,
        r=lora_rank, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        modules_to_save=mod_to_save)

    # Load pretrained model of original llama models
    if args.model_type =="llama1_pre" or args.model_type == "llama2_pre": 
        model = PeftModel.from_pretrained(llama_model, args.PEFT_PATH, is_trainable=False)
        model = get_peft_model(model, peft_config=new_peft_config)
    else:
        model = get_peft_model(llama_model, peft_config=new_peft_config)
        
    # Convert model parameters to bfloat16
    for param in model.parameters():
        if param.dtype == torch.float32 or param.dtype == torch.float16 :
            param.data = param.data.to(torch.bfloat16)
                
    model.print_trainable_parameters()
    model.to(device)

    step = 0
    wandb.init()
    # print("Starting Training for i2b2")
    # data = load_data(data_path + 'i2b2_2010/new_train.tsv', tokenizer=tokenizer, mode="train")
    # model = train_model(model, data, "i2b2_2010")
    # print("Completed Training")
    
    print("Starting Testing")
    logging.info("Starting Testing")
    data = load_data(data_path + 'i2b2_2010/new_test.tsv', tokenizer=tokenizer, mode="test")
    model = test_model(model, data, "i2b2_2010")
    print("Completed Testing for i2b2")
    wandb.finish()

    print("Completed supervised training and testing on GAD, euadr and i2b2 2010 benchmark datasets")

    
if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Supervised Fine-tuning of model")
    
    # Adding optional argument
    parser.add_argument("--model_type",required=True, help = "Model type")
    parser.add_argument("--PATH_TO_ORIGINAL_WEIGHTS", type=str,required=True,help="PATH TO ORIGINAL LLAMA WEIGHTS")
    parser.add_argument("--PEFT_PATH",type=str,help="PEFT_PATH")
    parser.add_argument("--mycache_dir",type=str, help="path to cache dir, required for llama2 ")
 
    # Read arguments from command line
    args = parser.parse_args()

    print(args)
    run_main(args)