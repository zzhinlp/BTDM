# BTDM: A Bi-directional Translating Decoding Model Based On Relational Embedding For Relational Triple Extraction
This repository contains the source code and dataset for the paper: **BTDM: A Bi-directional Translating Decoding Model Based On Relational Embedding For Relational Triple Extraction.**

## Overview

## Requirements

The main requirements are:

 - Keras==2.3.1
  - numpy==1.23.5
  - tensorflow==2.3.1
  - torch==1.12.1
  - transformers==4.25.1
  - tqdm

## Datasets

- [NYT*](https://github.com/weizhepei/CasRel/tree/master/data/NYT) and [WebNLG*](https://github.com/weizhepei/CasRel/tree/master/data/WebNLG)(following [CasRel](https://github.com/weizhepei/CasRel))
- [NYT](https://drive.google.com/file/d/1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N/view)(following [CopyRE](https://github.com/xiangrongzeng/copy_re))
- [WebNLG](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data)(following [ETL-span](https://github.com/yubowen-ph/JointER))


## Usage

**1. Get pre-trained BERT model for PyTorch**

Download [BERT-Base-Cased](https://huggingface.co/bert-base-cased/tree/main) which contains `pytroch_model.bin`, `vocab.txt` and `config.json`. Put these under `./pretrained`.

**2. Build Data**

Put our preprocessed datasets under `./datasets`.

**3. Train**

Specify the running mode and dataset at the command line

```python run.py ---train=train --dataset=NYT ```

**4. Evaluate**

Specify the running mode and dataset at the command line

```python run.py ---train=test --dataset=NYT ```
