'''
Created By: Anushuya Baidya
Date: 5/1/2024
'''
import argparse
import json

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer


class FineTuneLLM:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.id2label = {0: "anti-stereotype", 1: "stereotype", 2: "unrelated"}
        self.label2id = {'anti-stereotype': 0, 'stereotype': 1, 'unrelated': 2}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3, id2label=self.id2label, label2id=self.label2id
        )
        self.model = model.to(device)

        self.output_dir = "results/" + model_path

    def create_dataset(self, data):
        dataset_list = []

        # Iterate over intersentence data
        for intersentence in data['data']['intersentence']:
            bias_type = intersentence['bias_type']
            target = intersentence['target']
            context = intersentence['context']

            # Iterate over sentences within each intersentence
            for sentence_info in intersentence['sentences']:
                sentence = sentence_info['sentence']
                gold_label = sentence_info['gold_label']
                label = self.label2id[gold_label]
                d = {'context': context, 'sentence': sentence, 'bias_type': bias_type, 'target': target,
                     'label': label}
                dataset_list.append(d)

        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset_list))

        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

        train_data = datasets.Dataset.from_dict(train_data)
        test_data = datasets.Dataset.from_dict(test_data)

        # You can also specify the split name
        dataset_dict = DatasetDict({'train': train_data, 'test': test_data})
        return dataset_dict

    def tokenize_function(self, examples):
        return self.tokenizer(examples["context"], examples['sentence'], padding="max_length", truncation=True)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def train(self, dataset_dict):
        tokenized_data = dataset_dict.map(self.tokenize_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=True,
            output_dir=self.output_dir
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_model_path", required=True)

    args = parser.parse_args()

    print("args:", args)

    dataset_path = args.dataset_path
    model_path = args.model_path
    save_path = args.save_model_path

    fine_tune_llm = FineTuneLLM(model_path=model_path, device=device)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_dict = fine_tune_llm.create_dataset(data)
    fine_tune_llm.train(dataset_dict)

    fine_tune_llm.save_model(save_path=save_path)
