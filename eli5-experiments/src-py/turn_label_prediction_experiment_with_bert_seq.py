import sys
import os
import wandb

sys.path.insert(0, './src-py')
sys.path.insert(0, '../../model_sequence_labeling/lib/src')


import transformers
import datasets
import argparse
import torch
import json
from pathlib import Path
from datasets import load_dataset, Dataset, load_metric
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, default_data_collator,
                          PreTrainedModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from glob import glob

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=123)

import torch 
import random
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)


from multi_turn_bert import MultiTurnBert
from custom_dataset import CustomDataset, process_df
import main as seq_labeling

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_train_valid_splits(df, size=0.2):
    datasets = df.ds.unique()
    
    total_training_topics = []
    total_valid_topics = []
    for ds in datasets:
        topics = df[df.ds == ds].topic.unique()
        train_topics, valid_topics = train_test_split(topics, test_size=size)
        total_training_topics += train_topics.tolist()
        total_valid_topics += valid_topics.tolist()
    
    return total_training_topics, total_valid_topics

def run_fold_experiment(df, test_df, input_clm, label_clm, args, val_size=0.2):
    
    wandb.init(settings=wandb.Settings(start_method="fork"), project="explain-turn-label-pred-project", entity="milad-it", name='{}'.format(args.ckpt_dir.replace('/', '_').replace('..', '')))
    
    df['label'] = df[label_clm].apply(lambda labels: [int(x[2:4])-1 for x in labels]) #making labels parasable as integers
    test_df['label'] = test_df[label_clm].apply(lambda labels: [int(x[2:4])-1 for x in labels]) #making labels parasable as integers


    train_topics, valid_topics = get_train_valid_splits(df, val_size)

    model, eval_results = seq_labeling.run_experiment(args, 
                                         df[df.topic.isin(train_topics)], 
                                         df[df.topic.isin(valid_topics)], 
                                         test_df, input_clm, wandb=wandb)

    
    wandb.finish()
    
    return model, eval_results


def run_experiment(df, folds_dict, input_clm, label_clm, args, val_size=0.2):
    
    df = df.groupby('task_id').agg({'turn_text': lambda rows: list(rows),
                                    input_clm: lambda rows: list(rows),
                                    'ds': lambda rows: list(rows)[0],
                                    'topic': lambda rows: list(rows)[0],
                                    'topic_func_label': lambda rows: list(rows),
                                    'dlg_act_label': lambda rows: list(rows),
                                    'exp_act_label': lambda rows: list(rows)}).reset_index()
    
    ckpt_dir = args.ckpt_dir
    all_eval_results = []
    for i in range(0, 5):
        print('==== Running Fold {} ===== '.format(i))
        train_df = df[df.topic.isin(folds_dict['train']['5lvls'][i] + folds_dict['train']['eli5'][i])]
        test_df  = df[df.topic.isin(folds_dict['test']['5lvls'][i]  + folds_dict['test']['eli5'][i])]
        
        args.ckpt_dir = ckpt_dir + 'fold-{}'.format(i)
        model, eval_results = run_fold_experiment(train_df, test_df, input_clm, label_clm, args, val_size=val_size)
        print(eval_results)
        
        all_eval_results.append(eval_results)
        
    return all_eval_results


#Training models on the same five folds used for the quality prediction experiments -> so we test the performance of predicting quality of explanation on labels
#that are not ground-truth
def run_final_experiment(df, input_clm, label_clm, args, n_splits=5, val_size=0.2):
    ckpt_dir = args.ckpt_dir
    all_eval_results = []
    
    df = df.groupby('task_id').agg({'turn_text': lambda rows: list(rows),
                                    input_clm: lambda rows: list(rows),
                                    'ds': lambda rows: list(rows)[0],
                                    'topic': lambda rows: list(rows)[0],
                                    'topic_func_label': lambda rows: list(rows),
                                    'dlg_act_label': lambda rows: list(rows),
                                    'exp_act_label': lambda rows: list(rows)}).reset_index()
    
    #split the two corpora
    eli5_topics  = df[df.ds == 'eli5'].topic.unique()
    flvls_topics = df[df.ds == '5lvls'].topic.unique()

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    fold_idx = 0
    rmse_scores = []

    for fold in kfold.split(eli5_topics):
        print('==== Running Fold {} ===== '.format(fold_idx))
        train_topics = list(eli5_topics[fold[0]]) + list(flvls_topics)
        test_topics  = eli5_topics[fold[1]]
        
        train_df = df[df.topic.isin(train_topics)]
        test_df  = df[df.topic.isin(test_topics)]

        args.ckpt_dir = ckpt_dir + 'fold-{}'.format(fold_idx)
        model, eval_results = run_fold_experiment(train_df, test_df, input_clm, label_clm, args, val_size=val_size)
        print(eval_results)
        
        all_eval_results.append(eval_results)
        fold_idx+=1
        
    return all_eval_results

def load_ds(ds_path):
    df = pd.read_pickle(ds_path)

    #Aligning the 5-levels labels to eli5 ones
        
    #'(D06) To answer - Other' -> '(D06) Answer - Other'
    #'(D07) To provide agreement statement' -> '(D07) Agreement'
    #'(D08) To provide disagreement statement' -> '(D08) Disagreement'
    #'(D10) Other' -> '(D09) Other'
    #'(D09) To provide informing statement' -> (D10) To provide informing statement
    
    
    # (E10) Other -> (E09) Other 
    # (E09) Introducing Extraneous Information -> (E10) Introducing Extraneous Information
    
    df['exp_act_label'] = df.exp_act_label.apply(lambda x: '(E10) Other' if x == '(E09) Other' else x)
    df['exp_act_label'] = df.exp_act_label.apply(lambda x: '(E09) Introducing Extraneous Information' if x == '(E10) Introducing Extraneous Information' else x)

    df['dlg_act_label'] = df.dlg_act_label.apply(lambda x: '(D09) Other' if x == '(D10) Other' else x)
    df['dlg_act_label'] = df.dlg_act_label.apply(lambda x: '(D10) To provide informing statement' if x == '(D09) To provide informing statement' else x)
    
    df['dlg_act_label'] = df.dlg_act_label.apply(lambda x: '(D06) Answer - Other' if x == '(D06) To answer - Other' else x)
    df['dlg_act_label'] = df.dlg_act_label.apply(lambda x: '(D07) Agreement' if x == '(D07) To provide agreement statement' else x)
    df['dlg_act_label'] = df.dlg_act_label.apply(lambda x: '(D08) Disagreement' if x == '(D08) To provide disagreement statement' else x)
    

    df['turn_text_with_topic'] = df.apply(lambda row: {
                                        'author': row['turn_text']['author'], 
                                        'text'  : row['topic'].replace('_', ' ') + ' [SEP] ' +  row['turn_text']['text']
                                       } ,axis=1)

    return df

if __name__=="__main__":
    
    parser     = argparse.ArgumentParser()
    model_args = argparse.Namespace(turn_type='multi', pooling='cls', sp1_token='[EXPLAINER]', sp2_token='[EXPLAINEE]', bert_type='bert-base-uncased',
                          max_len=124, max_turns=15, dropout=0.1, device='cuda', learning_rate=2e-5, warmup_ratio=0.01,
                          batch_size=1, num_workers=2, num_epochs=5, num_classes=-1, ckpt_dir='../data/seq-labeling/model', planning=False, start_token='[START]')
    

    parser     = argparse.ArgumentParser()
    #parser.add_argument('label_clm', type=str)
    parser.add_argument('output_path', type=str) #'../data/bert_seq/mixed_ds_models/model/'
    #parser.add_argument('num_classes', type=int) #'../data/bert_seq/mixed_ds_models/model/'
    parser.add_argument('--ds', type=str, default='all')
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--input_clm', type=str, required=False, default="turn_text_with_topic")
    parser.add_argument('--final', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default='roberta-large')
    
    
    args = parser.parse_args()
    model_args.bert_type = args.model_name
    
    
    all_folds = json.load(open('../data/topic_folds.json'))
    
    #Loading and preparing data
    fivelvls_annotation_df = load_ds('../../data/five_levels_ds/annotation-results/MACE-measure/final_mace_predictions.pkl')
    eli5_annotation_df     = load_ds('../../data/eli5_ds/annotation-results/MACE-measure/final_mace_predictions_training.pkl')
    
    fivelvls_annotation_df['ds'] = ['5lvls'] * len(fivelvls_annotation_df)
    eli5_annotation_df['ds'] = ['eli5'] * len(eli5_annotation_df)
    dlgs_df = pd.concat([fivelvls_annotation_df, eli5_annotation_df])
    
    for label_clm, num_classes in [('dlg_act_label', 10), ('exp_act_label', 10), ('topic_func_label', 4)]:
        for ds in ['5lvls', 'eli5', 'all']:

            if ds != 'all':
                working_dlgs_df = dlgs_df[dlgs_df.ds == ds].copy()
            else:
                working_dlgs_df = dlgs_df.copy()

            if len(working_dlgs_df) == 0:
                print('ds specified doesnt exist...')
                exit()

            output_path = '{}/{}/{}_prediction/{}_models/model/'.format(args.output_path, model_args.bert_type, label_clm, ds)

            print('Training on {} with size {}'.format(ds, len(working_dlgs_df)))
            print('Output path {}'.format(output_path))

            model_args.learning_rate = 2e-6
            model_args.batch_size = 2 #2
            model_args.max_turns  = 56
            model_args.max_len    = 256
            model_args.num_epochs = args.num_train_epochs
            model_args.num_classes= num_classes
            model_args.pooling    = 'cls'
            model_args.planning   = False
            model_args.ckpt_dir   = output_path


            print(args)
            print(model_args)


            if args.final == True:
                print('Training final model')
                eval_results = run_final_experiment(working_dlgs_df, args.input_clm, label_clm, model_args)
            else:
                eval_results = run_experiment(working_dlgs_df, all_folds, args.input_clm, label_clm, model_args)

            json.dump(eval_results, open('{}/eval_results.json'.format(output_path), 'w'))

            print(eval_results)

