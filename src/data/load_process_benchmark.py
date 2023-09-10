"""
    Load and process benchmark datasets

    Author: Lathashree Harisha
"""


import os
import re
import torch

import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    filename='application.log', filemode='w')

  

class BenchMark_Dataset(Dataset):
    """ Custom class for Dataset Object"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

class EUADR_Dataset():
    """ Load and process EUADR dataset"""

    def __init__(self, folder_path, tokenizer, validation_split=0.2):
        super().__init__()
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.validation_split = validation_split
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        print(f"Loading EUADR data from folder -- {self.folder_path}"
            f" with validation split {self.validation_split}")
        self.load_data()
  
    def load_data(self):
        """ Load and process EUADR dataset"""

        for root, _, all_files in os.walk(self.folder_path, topdown=False):
            for file in all_files:
                filename = os.path.join(root, file)
                if file.startswith("train.tsv"):
                    train_df = pd.read_csv(filename,header=None, delimiter='\t', names=['sentence', 'label']) 
                    self.train_data = pd.concat([self.train_data, train_df], ignore_index=True)

                if file.startswith("dev.tsv"):
                    self.file_path = os.path.join(self.folder_path, file)
                    val_df = pd.read_csv(filename,header=None, delimiter='\t', names=['sentence', 'label'])
                    self.val_data = pd.concat([self.val_data, val_df], ignore_index=True)
                
                if file.startswith("test.tsv"):
                    self.file_path = os.path.join(self.folder_path, file)
                    test_df = pd.read_csv(filename,header=None, delimiter='\t', names=['sentence', 'label'])
                    self.test_data = pd.concat([self.test_data, test_df], ignore_index=True)
        
        if not self.val_data.size:
            self.train_data, self.val_data = train_test_split(self.train_data, test_size=self.validation_split, random_state=42)

        print("Train data size: {}".format(self.train_data.size))
        print("Validation data size: {}".format(self.val_data.size))
        print("Test data size: {}".format(self.test_data.size))
        logging.info("EUADR dataset: Train data size: {}; Validation data size {}, Test data size {}".format(self.train_data.size, self.val_data.size, self.test_data.size, self.test_data.size))

    def process_data(self):
        """ Process EUADR dataset"""
        train_encoding = self.tokenizer(self.train_data['sentence'].tolist(), truncation=True, padding=True)
        train_labels = self.train_data['label'].tolist()
        test_encoding = self.tokenizer(self.test_data['sentence'].tolist(), truncation=True, padding=True)
        test_labels = self.test_data['label'].tolist()
        val_encoding = self.tokenizer(self.val_data['sentence'].tolist(), truncation=True, padding=True)
        val_labels = self.val_data['label'].tolist()
        return BenchMark_Dataset(train_encoding,train_labels),  BenchMark_Dataset(val_encoding,val_labels), BenchMark_Dataset(test_encoding,test_labels)

class GAD_Dataset():
    """ Load and process GAD dataset"""

    def __init__(self, folder_path, tokenizer, validation_split=0.2):
        super().__init__()
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.validation_split = validation_split
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        print(f"Loading GAD data from folder -- {self.folder_path}"
            f" with validation split {self.validation_split}")
        
        self.load_process_data()
    
    def load_process_data(self):
        """ Load and process EUADR dataset"""

        for root, _, all_files in os.walk(self.folder_path, topdown=False):
            for file in all_files:
                filename = os.path.join(root, file)
                if file.startswith("train.tsv"):
                    train_df = pd.read_csv(filename,header=None, delimiter='\t', names=['sentence', 'label']) 
                    self.train_data = pd.concat([self.train_data, train_df], ignore_index=True)

                if file.startswith("dev.tsv"):
                    self.file_path = os.path.join(self.folder_path, file)
                    val_df = pd.read_csv(filename,header=None, delimiter='\t', names=['sentence', 'label'])
                    self.val_data = pd.concat([self.val_data, val_df], ignore_index=True)
                
                if file.startswith("test.tsv"):
                    self.file_path = os.path.join(self.folder_path, file)
                    test_df = pd.read_csv(filename,header=None, delimiter='\t', names=['sentence', 'label'])
                    self.test_data = pd.concat([self.test_data, test_df], ignore_index=True)
        
        if not self.val_data.size:
            self.train_data, self.val_data = train_test_split(self.train_data, test_size=self.validation_split, random_state=42)

        print("Train data size: {}".format(self.train_data.size))
        print("Validation data size: {}".format(self.val_data.size))
        print("Test data size: {}".format(self.test_data.size))
        logging.info("GAD dataset: Train data size: {}; Validation data size {}, Test data size {}".format(self.train_data.size, self.val_data.size, self.test_data.size, self.test_data.size))
    

class i2b2_Dataset():
    """ Loading i2b2 dataset """

    label2id = {'TrIP': 0, 'TrWP': 1,'TrCP': 2,'TrAP': 3,'TrNAP': 4,'PIP': 5,'TeRP': 6,'TeCP': 7}

    def __init__(self, folder_path, tokenizer, validation_split=0.3):
            super().__init__()
            self.folder_path = folder_path
            self.tokenizer = tokenizer
            self.validation_split = validation_split
            self.train_data = pd.DataFrame()
            self.val_data = pd.DataFrame()
            self.test_data = pd.DataFrame()
            self.BETH = folder_path + 'beth/'
            self.PARTNERS = folder_path + 'partners/'
            self.TRAIN_DATA = [self.BETH, self.PARTNERS]
            print(f"Loading i2b2 data from folder -- {self.folder_path}"
                f" with validation split {self.validation_split}")
            self.load_process_data()
    

    def load_process_data(self):
            extract_concept_pattern = r"c=\"(.+?)\" (\d+):(\d+) (\d+):(\d+)"
            extract_rel_pattern = r"r=\"(.+?)\""
            sentences = []
            labels = []

            # Iterate over training folders: BETH, PARTNERS
            for provider in self.TRAIN_DATA:
                all_files = os.walk(provider)
                for root, _, all_files in os.walk(provider, topdown=False):
                    for name in all_files:
                        file = os.path.join(root, name)
                        if file.endswith('.rel'):                
                            unique_file_match = re.search(r'([^/]+)\.rel$', file)
                            text_file_name = os.path.join(provider,'txt',unique_file_match.group(1) + '.txt')
                            with open(text_file_name, 'r') as text_file,  open(file, 'r') as rel_file:
                                text_lines = text_file.readlines()
                                for rel_line in rel_file:
                                    first_part, sec_part, third_part = rel_line.split('||')
                                    c1_match = re.findall(extract_concept_pattern, first_part)[0]
                                    c1_start_line = int(c1_match[1])
                                    c1_start_token = int(c1_match[2])
                                    c1_end_token = int(c1_match[4])
                                    c2_match = re.findall(extract_concept_pattern, third_part)[0]
                                    c2_end_token = int(c2_match[4])
                                    c2_start_token = int(c2_match[2])
                                    r_match = re.findall(extract_rel_pattern, sec_part)[0]
                                    position = c1_start_line-1
                                    sent = text_lines[position]

                                    sent_list = sent.split(" ")
                                    sent_list[c1_start_token] = '@'+sent_list[c1_start_token]
                                    sent_list[c1_end_token] = sent_list[c1_end_token]+'$'
                                    sent_list[c2_start_token] = '@'+sent_list[c2_start_token]
                                    sent_list[c2_end_token] = sent_list[c2_end_token]+'$'
                                    sent = " ".join(sent_list)

                                    sent = sent.replace(c1_match[0], '@'+c1_match[0]+'$')
                                    sent = sent.replace(c2_match[0], '@'+c2_match[0]+'$')
                                    label = self.label2id[r_match]

                                    sentences.append(sent)
                                    labels.append(label)

            self.train_data = pd.DataFrame({'sentence': sentences, 'label': labels})
            self.train_data, test_val_data = train_test_split(self.train_data, test_size=self.validation_split, random_state=42)
            self.val_data, self.test_data = train_test_split(test_val_data, test_size=0.35, random_state=42)

            print("Train data size: {}".format(self.train_data.size))
            print("Validation data size: {}".format(self.val_data.size))
            print("Test data size: {}".format(self.test_data.size))
            logging.info("i2b2 dataset: Train data size: {}; Validation data size {}, Test data size {}".format(self.train_data.size, self.val_data.size, self.test_data.size, self.test_data.size))


class n2c2_Dataset():
    """ Loads n2c2 2018 dataset"""

    label2id = {'Strength-Drug':0, 'Form-Drug': 1, 'Dosage-Drug':2, 'Frequency-Drug':3, 'Route-Drug':4, 'Duration-Drug':5, 'Reason-Drug':6, 'ADE-Drug':7}
    
    def __init__(self, folder_path, tokenizer, validation_split=0.2):
        super().__init__()
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.validation_split = validation_split
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        print(f"Loading n2c2 data from folder -- {self.folder_path}"
            f" with validation split {self.validation_split}")
        self.load_data()


    def load_data(self):
        train_folder = self.folder_path + 'processed_train/'
        test_folder = self.folder_path + 'processed_test/'

        self.train_data = pd.read_csv(train_folder + 'train_re.txt', names=['sentence', 'label'])
        self.test_data = pd.read_csv(test_folder + 'test_re.txt', names=['sentence', 'label'])

        self.train_data['label'] = self.train_data['label'].map(self.label2id)
        self.test_data['label'] = self.test_data['label'].map(self.label2id)

        self.train_data, self.val_data = train_test_split(self.train_data, test_size=self.validation_split, random_state=42)

        print("Train data size: {}".format(self.train_data.size))
        print("Validation data size: {}".format(self.val_data.size))
        print("Test data size: {}".format(self.test_data.size))
        logging.info("n2c2 dataset: Train data size: {}; Validation data size {}, Test data size {}".format(self.train_data.size, self.val_data.size, self.test_data.size, self.test_data.size))


def benchmark_dataload_main(dataset_name, tokenizer):
    """
        Main function to load and process benchmark datasets

        Args:
            dataset_name: Name of the dataset
            tokenizer: Tokenizer object

        Returns:
            dataset: Dataset object

    """
    # print("Loding benchmark datasets...")
    if dataset_name == 'euadr':
        eu_data = EUADR_Dataset(folder_path='../../datasets/euadr/', tokenizer=tokenizer)
        return eu_data.process_data()
    elif dataset_name == 'gad':
        gad_data = GAD_Dataset(folder_path='../../datasets/GAD/', tokenizer=tokenizer)
        return gad_data
    elif dataset_name == 'i2b2':
        i2b2_data = i2b2_Dataset(folder_path='../../datasets/i2b2_2010/', tokenizer=tokenizer)
        return i2b2_data
    elif dataset_name == 'n2c2':
        n2c2_data = n2c2_Dataset(folder_path='../../datasets/n2c2_2018/', tokenizer=tokenizer)
        return n2c2_data

if __name__ == "__main__":
    benchmark_dataload_main('euadr', tokenizer=None)