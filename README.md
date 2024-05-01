# Empowering Large Language Models to Understand Bias

## Project Description
The project aims to investigate whether Large Language Models (LLMs) can effectively understand and mitigate bias. Bias is a critical issue in natural language processing (NLP), and it is essential to ensure that LLMs have the capability to comprehend and address bias appropriately. This project explores the potential of LLMs to recognize and mitigate bias through fine-tuning on bias-related datasets.

## Code Overview
The project includes two main files:

### 1. fine_tune_llm.py
This file contains the code for fine-tuning a LLM model on the StereoSet dataset. It can be run from the command line using the following command:

```commandline
python src/fine_tune_llm.py --dataset_path "Path/to/dataset/file" --model_path "bert-base-uncased" --save_model_path "path/to/save/finetuned/model"
```
You can change the `model_path` parameter to specify any pre-trained LLM model you wish to fine-tune. Once the model is fine-tuned, it will be saved in the results folder.

### 2. evaluate_llm.py
This file allows you to evaluate any LLM model, whether it is a pre-trained model or your own fine-tuned model. It can be run from the command line using the following command:

```commandline
python src/evaluate_llm.py --dataset_path "Path/to/dataset/file" --model_path "results/finetuned_model"
```
You need to specify the `dataset_path` parameter to provide the path to the dataset file. The `model_path` parameter should point to the location of the model to be evaluated.

## Requirements
- Python 3.x
- PyTorch
- Transformers library


