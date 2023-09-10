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
│   │   ├── train.tsv
│   │   ├── test.tsv
│   ├──GAD/
│   │   ├── train.tsv
│   │   ├── test.tsv
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

The datasets used in this project are as follows:
1. Clincial datasets: i2b2 2010 and n2c2 2018. Due to usage agreements, we have not shared the datsets, please use https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/ to gain access to the datasets.
2. Biomedical datasets: GAD and EUADR. The datasets are available in the datasets folder.

## Models

The models used in this project are LLaMA 1 and LLaMA 2.

## Set up and Installation

### 1. Clone the repository

```bash
git clone
```

### 2. Install the requirements

```bash
pip install -r requirements.txt
```

### 3. Download the models

```bash


```

### 4. Running the DAP

```bash
python src/models/dap.py
```

### 5. Running the Supervised Fine-tuning

```bash
python src/models/lra.py
```


## Results


## Acknowledgements

This project is based on the following open-source projects for secondary development.

- https://github.com/ymcui/Chinese-LLaMA-Alpaca/
- https://github.com/uf-hobi-informatics-lab/NLPreprocessing
- https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction


## Main libraries used
- https://github.com/huggingface/transformers
- https://github.com/huggingface/peft
- https://github.com/huggingface/accelerate

