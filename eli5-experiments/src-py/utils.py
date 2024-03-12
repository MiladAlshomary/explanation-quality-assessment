import sys

sys.path.insert(0, '../../../third-party/hierarchical-transformers/')
sys.path.insert(0, '../../models/')

import torch
from transformers import LongformerTokenizerFast
from transformers import LongformerTokenizer

from hatformer import MyHATForSequenceClassification, HATConfig, HATTokenizer
from longformer import LongformerForSequenceClassification, LongformerConfig
from models.hat import HATForSequenceClassification


from transformers import (AutoTokenizer, AutoModelForSequenceClassification, default_data_collator,
                          PreTrainedModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from language_modelling.data_collator import DataCollatorForDocumentClassification

from datasets import load_dataset, Dataset, load_metric
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import json
import wandb

import random
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=123)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
exp_act_padding_token = 10

print('Device: ', device)


hat_former_name = 'kiddothe2b/hierarchical-transformer-base-4096'
longformer_model_name = "allenai/longformer-base-4096"


def get_hatformer_model(extra_encoder_configs, model_path=None):
    model_revision="main"
    pooling="max" #Which pooling method to use (max, cls, attentive)

    model_path = hat_former_name if model_path == None else model_path
    print("Loading model: {}".format(model_path))
    config    = HATConfig.from_pretrained(model_path, num_labels=1, finetuning_task="document-classification", revision="main", use_auth_token=None)
    config.num_labels=1
    config.max_sentence_size=32
    config.max_sentence_length=128
    config.max_sentences = 32
    tokenizer = HATTokenizer.from_pretrained(hat_former_name, do_lower_case=False, revision=model_revision, use_auth_token=None)
    model     = MyHATForSequenceClassification.from_pretrained(model_path, pooling=pooling, config=config, revision=model_revision, use_auth_token=None, extra_encoders_configs=extra_encoder_configs)
    
    return config, model, tokenizer

def get_longformer_model(extra_encoder_configs, model_path=None):
    
    model_path = longformer_model_name if model_path == None else model_path
    print("Loading model: {}".format(model_path))
    tokenizer = LongformerTokenizer.from_pretrained(longformer_model_name)
    config = LongformerConfig.from_pretrained(model_path)
    config.num_labels = 1
    
    model = LongformerForSequenceClassification.from_pretrained(model_path, config=config, extra_encoders_configs=extra_encoder_configs).to(device)
    
    return config, model, tokenizer

def load_results(fold):
    import glob
    scores = []
    for f in glob.glob('{}/*/eval_results.json'.format(fold)):
        res = json.load(open(f))
        scores.append(res['eval_rmse'])
    return scores, np.mean(scores)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def get_dlg_exp_moves(dlg_exp_moves):
    dlgs_moves = [[int(turn) for turn in dlg] for dlg in dlg_exp_moves]
    #paddig
    max_len    = max([len(x) for x in dlgs_moves])
    pad_token  = 0
    dlgs_moves = np.array([dlg_move[:max_len] if len(dlg_move) > max_len else dlg_move + [pad_token] * (max_len - len(dlg_move)) for dlg_move in dlgs_moves])
    dlgs_moves = np.expand_dims(dlgs_moves, axis=1) # to get something like [[1], [2], ...]
    return dlgs_moves

def preprocess_hat_function(tokenizer, examples, max_seq_length, padding):
        # Tokenize the texts
        batch = tokenizer(
            examples["input_texts"],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )

        batch = tokenizer.pad(
            batch,
            padding=padding,
            max_length=max_seq_length,
            pad_to_multiple_of=max_seq_length,
        )

        #batch["label_ids"] = [[1.0 if label in labels else 0.0 for label in label_list] for labels in examples["labels"]]
        batch["label_ids"] = [float(label) for label in examples["labels"]]

        #adding flows represenations
        flow_ids = []
        if "exp_act_label" in examples:
            print('adding exp_act_label to the flows')
            flow_ids.append(get_dlg_exp_moves(examples['exp_act_label']))
        if "dlg_act_label" in examples:
            print('adding dlg_act_label to the flows')
            flow_ids.append(get_dlg_exp_moves(examples['dlg_act_label']))
        if "topic_func_label" in examples:
            print('adding topic_func_label to the flows')
            flow_ids.append(get_dlg_exp_moves(examples['topic_func_label']))

        if flow_ids != []:
            batch_flow_ids = np.concatenate(flow_ids, axis=1)
            
            #construct the mask
            #print(batch_flow_ids.shape)
            mask = np.minimum.reduce(batch_flow_ids, axis=1) # Find indices that has zero value (the padding token)
            #print(mask)
            #print(mask.shape)
            batch['flow_ids'] = batch_flow_ids.tolist()
            batch['flow_ids_mask'] = np.invert(mask.astype(bool)).tolist()
            
            #print(batch['flow_ids'][:3])
            #print(batch['flow_ids_mask'][:3])
            
        return batch

def preprocess_function(tokenizer, df, input_clm='turn_text', extra_exp_moves=False, extra_exp_types=False, global_attention=False):
    print('Using input_clm={}'.format(input_clm))
    
    tokenized_corpus = []
    encoded_corpus = []
    for idx, row in df.iterrows():

        tokenized_turns = [{'tokens': tokenizer.tokenize(turn) + [tokenizer.sep_token]} 
                               for turn in row[input_clm]]

        encoded_dialoge = {
                'token_ids': tokenizer.encode_plus([token for turn in tokenized_turns for token in turn['tokens']], truncation=True, padding='max_length')
        }

        encoded_dialoge = {
            'input_ids': encoded_dialoge['token_ids']['input_ids'],
            'attention_mask': encoded_dialoge['token_ids']['attention_mask']
        }

        tokenized_corpus.append(tokenized_turns)
        encoded_corpus.append(encoded_dialoge)

    output = {
        'input_ids': torch.tensor(np.array([x['input_ids'] for x in encoded_corpus])).to(device),
        'attention_mask': torch.tensor(np.array([x['attention_mask'] for x in encoded_corpus])).to(device),
        #'global_attention_mask': np.array([x['attention_mask'] for x in encoded_corpus]),
        'labels': torch.tensor(df.labels.values.astype(float)).to(device)
    }

    #adding flows represenations
    flow_ids = []
    if "exp_act_label" in df.columns:
        print('adding exp_act_label to the flows')
        flow_ids.append(get_dlg_exp_moves(df['exp_act_label']))
    if "dlg_act_label" in df.columns:
        print('adding dlg_act_label to the flows')
        flow_ids.append(get_dlg_exp_moves(df['dlg_act_label']))
    if "topic_func_label" in df.columns:
        print('adding topic_func_label to the flows')
        flow_ids.append(get_dlg_exp_moves(df['topic_func_label']))

    if flow_ids != []:
        batch_flow_ids = np.concatenate(flow_ids, axis=1)
            
        #construct the mask
        #print(batch_flow_ids.shape)
        mask = np.minimum.reduce(batch_flow_ids, axis=1) # Find indices that has zero value (the padding token)
        #print(mask)
        #print(mask.shape)
        output['flow_ids_mask'] = torch.tensor(np.invert(mask.astype(bool)))
        output['flow_ids'] = torch.tensor(batch_flow_ids)
    
    return output

def load_and_prepare_df(path, quality_df_path):
    eli5_df  = pd.read_pickle(path)
    eli5_dlg_quality_df = pd.read_csv(quality_df_path)
    quality_scores = pd.Series(eli5_dlg_quality_df.rating_label.values, index = eli5_dlg_quality_df.task_id).to_dict()
    
    eli5_df = eli5_df.groupby('task_id').agg({'turn_text': lambda rows: list(rows),
                                          'topic': lambda rows: list(rows)[0],
                                          'topic_func_label': lambda rows: list(rows),
                                          'dlg_act_label': lambda rows: list(rows),
                                          'exp_type_label' : lambda rows: list(rows),
                                          'exp_act_label': lambda rows: list(rows)}).reset_index()

    eli5_df['labels'] = eli5_df.task_id.apply(lambda x: quality_scores[x])
    eli5_df['input_texts'] = eli5_df.turn_text.apply(lambda row: [x['text'] for x in row])

    eli5_df['exp_act_label'] =  eli5_df['exp_act_label'].apply(lambda row: [int(x[2:4]) for x in row])
    eli5_df['dlg_act_label'] =  eli5_df['dlg_act_label'].apply(lambda row: [int(x[2:4]) for x in row])
    eli5_df['topic_func_label'] =  eli5_df['topic_func_label'].apply(lambda row: [int(x[2:4]) for x in row])

    print('Maximum seq num:', max([len(x) for x in eli5_df.input_texts.tolist()]))
    print('Maximum seq len:', max([len(turn.split(' ')) for turns in eli5_df.input_texts.tolist() for turn in turns]))
    print('Data size:', len(eli5_df))
    return eli5_df