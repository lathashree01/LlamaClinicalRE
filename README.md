# A Unified Framework for Clinical Relation Extraction

### Introduction

The work presents the development of a Clinical Relation Extraction framework
built upon Large Language Models (LLMs). The primary objective of this work
is to harness the exceptional capabilities of LLMs, specifically the foundational
language model known as LLaMA, for effective clinical domain tasks. The project
comprises a two-stage approach, beginning with the adaptation of LLaMA to
the clinical domain. The resulting clinical LLaMA models are further fine-tuned
on predetermined evaluation benchmark datasets in a supervised setting. Our
training methodologies use a parameter-efficient fine-tuning technique called
Low-Rank Adaptation. The comprehensive evaluation and analysis conducted on
both clinical and biomedical benchmark datasets helped us highlight the strengths
and limitations of our clinical LLaMA models.


### Our Framework

![Framework](./visualisation/framework.png)


## Code structure

The code directory is structured as follows:
```
LLamaClincialRE/
├── datasets/
│   ├── euadr/
│   ├──GAD/
│   ├── i2b2_2010/   
│   └──n2c2_2018/ 
├── notebooks/
├── scripts/
├── models/
├── src/
│   ├── data/
│   └── models/
├── visualisation/
├── requirements/
├── .gitignore
|── app.log
├── final_report.pdf
└── README.md
```

## Datasets

Pretraining datasets used in this project are as follows:
- MIMIC-III: https://physionet.org/content/mimiciii/1.4/

The evaluation datasets used in this project are as follows:
- Clincial datasets: i2b2 2010 and n2c2 2018. Due to usage agreements, we have not shared the datsets, please use https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/ to gain access to the datasets.
- Biomedical datasets: GAD and EUADR. The datasets are available in the datasets folder (preprocessed versions from https://github.com/dmis-lab/biobert ).

## Models

The base models used in this project are LLaMA 1 and LLaMA 2.

## Set up and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Lathashree01/LlamaClinicalRE.git
```

### 2. Install the requirements
The requirements are divided into three environments:
- clm_requirements.txt: Requirements for the DAP of LLaMA models
- finetune_requirements.txt: Requirements for the fine-tuning the clincial LLaMA models 
- ufcode_requirements.txt: Requirements for the UF code: used for supervised fine-tuning on n2c2 2018 dataset

Example:
```bash
cd requirements
pip install -r clm_requirements.txt
```

### 3. Download the models
Please download LLaMA models from https://huggingface.co/meta-llama

### 4. Running the DAP

```bash
sh src/models/run_pt.sh
```

### 5. Running the Supervised Fine-tuning

```bash
python src/models/lra.py
```

### Results

Please refere to chapter 4 of the final report for the results.


## Acknowledgements

This project is based on the following open-source projects for secondary development.

- https://github.com/ymcui/Chinese-LLaMA-Alpaca/
- https://github.com/uf-hobi-informatics-lab/NLPreprocessing
- https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction


## Main libraries used
- https://github.com/huggingface/transformers
- https://github.com/huggingface/peft
- https://github.com/huggingface/accelerate

