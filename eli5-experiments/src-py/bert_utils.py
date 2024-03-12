import transformers
import datasets

import torch
import json
from pathlib import Path
from datasets import load_dataset, Dataset, load_metric
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, default_data_collator,
                          PreTrainedModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput

from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from glob import glob

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=123)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def majority_class(df):
    topics = df.topic.unique()
    for topic in topics:
        training_df = df[df.topic != topic]
        #compute the majority class for each label
        l = len(df[df.topic == topic])
        df.loc[df.topic == topic, 'topic_func_maj_pred'] = [training_df.topic_func_label.mode()] * l
        df.loc[df.topic == topic, 'dlg_act_maj_pred']    = [training_df.dlg_act_label.mode()] * l
        df.loc[df.topic == topic, 'exp_act_maj_pred']    = [training_df.exp_act_label.mode()] * l
    
    return df

def eval_preds(df, models_names, gt_clms, pred_clms):
    results_table = []
    for label in zip(gt_clms, pred_clms, models_names):
        ground_truths = df[label[0]].tolist()
        predictions   = df[label[1]].tolist()
        model_name = label[2]
        
        class_names = df[label[0]].unique()

        prc_scores = precision_score(ground_truths, predictions, average=None, labels=class_names)
        rec_scores = recall_score(ground_truths, predictions, average=None, labels=class_names)
        f1_scores  = f1_score(ground_truths, predictions, average=None, labels=class_names)
        
        macro_prc_scores = precision_score(predictions, ground_truths, average='macro', labels=class_names)
        macro_rec_scores = recall_score(predictions, ground_truths, average='macro', labels=class_names)
        macro_f1 = f1_score(predictions, ground_truths, average='macro', labels=class_names)
        
        scores ={}
        for i, c in enumerate(class_names):
            scores[c] = {'prec': round(prc_scores[i],2), 'recall': round(rec_scores[i],2), 'f1': round(f1_scores[i],2)}
        
        scores['Macro AVG.'] = {'prec': round(macro_prc_scores,2), 'recall': round(macro_rec_scores,2), 'f1': round(macro_f1,2)}
        
        results_table.append([model_name, label[0], scores])
    
    return results_table

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    f1score = f1_score(predictions, labels, average='macro')
    return {'f1-score': f1score}

def preprocess_function(examples, input_clm='turn_text'):
    # Tokenize the texts
    texts = [x['text'] for x in examples[input_clm]]
    result = tokenizer(texts, truncation=True, padding='max_length')
    
    if 'labels' in examples:
        result['labels'] = examples['labels']
        
    return result

def bert_pred_labels(model_path, df, label_clm, output_clm, input_clm):
    label_dictionary = {int(l[2:4])-1 : l for l in  df[label_clm].unique()}
            
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    eval_dataset = Dataset.from_pandas(df)
    eval_dataset = eval_dataset.map(lambda examples: preprocess_function(examples, input_clm), batched=True)
    eval_dataset = eval_dataset.remove_columns(df.columns.tolist() + ['__index_level_0__'])

    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator)
    all_predictions = []
    for step, batch in enumerate(eval_dataloader):
        batch = {x[0]: x[1].cuda() for x in batch.items()}
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions+=[x.item() for x in predictions]

    df[output_clm] = all_predictions
        
    return df

def train_model(model, tokenizer, output_dir, train_ds, valid_ds, test_ds, num_train_epochs=5, eval_steps=500, lr=2e-6, batch_size=4):

    args = TrainingArguments(
        output_dir= output_dir,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='f1-score'
    )

    
    multi_trainer =Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=lambda x: compute_metrics(x),
        tokenizer=tokenizer
    )
    
    multi_trainer.train()
    
    model.save_pretrained(output_dir)
    eval_results = multi_trainer.evaluate(test_ds)
    
    return model, eval_results
