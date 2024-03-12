import sys
import os
import wandb

sys.path.insert(0, './src-py')
sys.path.insert(0, '../../../third-party/hierarchical-transformers/')

import argparse
import json
import pandas as pd
import numpy as np
import glob 
from tabulate import tabulate
import math
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from models.hat import HATConfig, HATTokenizer, HATForSequenceClassification
from models.longformer import LongformerTokenizer, LongformerModelForSequenceClassification
from language_modelling.data_collator import DataCollatorForDocumentClassification
from datasets import load_dataset, Dataset, load_metric
from sklearn.metrics import f1_score

from utils import *


hat_former_name = 'kiddothe2b/hierarchical-transformer-base-4096'
longformer_model_name = "allenai/longformer-base-4096"

hat_former_feats_clms_encoders = {
    "exp_act_label": {'num_tokens':11, 'flow_model_hidden_size': 128, 'nhead':8, 'nlayers':12},
    "dlg_act_label": {'num_tokens':11, 'flow_model_hidden_size': 128, 'nhead':1, 'nlayers':12},
    "topic_func_label": {'num_tokens':5, 'flow_model_hidden_size': 128, 'nhead':4, 'nlayers':3}
}

long_former_feats_clms_encoders = {
    "exp_act_label": {'num_tokens':11, 'flow_model_hidden_size': 128, 'nhead':4, 'nlayers':6},
    "dlg_act_label": {'num_tokens':11, 'flow_model_hidden_size': 128, 'nhead':1, 'nlayers':12},
    "topic_func_label": {'num_tokens':5, 'flow_model_hidden_size': 128, 'nhead':8, 'nlayers':12}
}
    
def train_and_evaluate_single_fold(model_type, output_dir, df, wandb_run_name, test_df=None, input_clm='turn_text', num_train_epochs=5, eval_steps=500, lr=2e-5, batch_size=4, max_seq_length=4096, extra_encoder_configs=[], only_eval=False, model_path=None):


    if model_type == 'hatformer':
        config, model, tokenizer = get_hatformer_model(extra_encoder_configs, model_path=model_path)
    elif model_type == 'longformer':
        config, model, tokenizer = get_longformer_model(extra_encoder_configs, model_path=model_path)
    else:
        print('No model type identified..')
        return
    

    #split by topic
    topics  = df.topic.unique()
    train_topics, valid_topics  = train_test_split(topics, test_size=0.2, shuffle=True, random_state=123)
    

    wandb.init(settings=wandb.Settings(start_method="fork"), project="exp-quality-project", entity="milad-it", name='{}'.format(wandb_run_name))

    train_df = df[df.topic.isin(train_topics)]
    valid_df = df[df.topic.isin(valid_topics)]
    test_df  = valid_df if test_df is None else test_df

    #balance the data
    train_df, y = ros.fit_resample(train_df, train_df['labels'])
    train_df['labels'] = y

    training_ds = Dataset.from_pandas(train_df)
    valid_ds    = Dataset.from_pandas(valid_df)
    test_ds     = Dataset.from_pandas(test_df)

    if model_type == 'hatformer':
        training_ds = training_ds.map(lambda examples: preprocess_hat_function(tokenizer, examples, max_seq_length, False), 
                            batched=True, load_from_cache_file=False, remove_columns=['labels'])
        valid_ds    = valid_ds.map(lambda examples: preprocess_hat_function(tokenizer, examples, max_seq_length, False), 
                            batched=True, load_from_cache_file=False, remove_columns=['labels'])
        test_ds     = test_ds.map(lambda examples: preprocess_hat_function(tokenizer, examples, max_seq_length, False), 
                            batched=True, load_from_cache_file=False, remove_columns=['labels'])
    else:
        training_ds = Dataset.from_dict(preprocess_function(tokenizer, train_df, input_clm=input_clm))
        valid_ds = Dataset.from_dict(preprocess_function(tokenizer, valid_df, input_clm=input_clm))
        test_ds = Dataset.from_dict(preprocess_function(tokenizer, test_df, input_clm=input_clm))

    print('Training {}, Valid {}, and Test {}'.format(len(training_ds), len(valid_ds), len(test_ds)))

    args = TrainingArguments(
            output_dir= '{}/{}/{}-fold'.format(output_dir, wandb_run_name, 0),
            overwrite_output_dir=True,
            evaluation_strategy = "steps",
            save_strategy = "steps",
            logging_strategy="steps",
            save_total_limit=5,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            eval_steps=eval_steps,
            logging_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model='rmse'
        )


    trainer = Trainer(
        model,
        args,
        train_dataset=training_ds,
        eval_dataset=valid_ds,
        compute_metrics=lambda x: compute_metrics(x),
        tokenizer=tokenizer
    )

    if only_eval == False:
        trainer.train()

    eval_results = trainer.evaluate(test_ds)

    if only_eval == False: #only save results when we are training also
        model.save_pretrained('{}/{}/{}-fold'.format(output_dir, wandb_run_name, 0))
        test_ds.to_json('{}/{}/{}-fold/test_set.json'.format(output_dir, wandb_run_name, 0))
        json.dump(eval_results, open('{}/{}/{}-fold/eval_results.json'.format(output_dir, wandb_run_name, 0), 'w'))

    wandb.finish()
        
    return eval_results['eval_rmse']


def cross_fold_training_and_evaluation(model_type, output_dir, df, wandb_run_name, input_clm='turn_text', num_train_epochs=5, eval_steps=500, lr=2e-5, batch_size=4, max_seq_length=4096, extra_encoder_configs=[], n_splits=3, only_eval=False, model_path=None):


    rmse_scores = []

    # #split by topic
    # topics  = df.topic.unique()
    # train_topics, test_topics  = train_test_split(topics, test_size=0.2, shuffle=True, random_state=123)
    # train_topics, valid_topics = train_test_split(train_topics, test_size=0.2, shuffle=True, random_state=123)
    
    #split the two corpora
    topics  = df.topic.unique()

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    fold_idx = 0
    rmse_scores = []
    for fold in kfold.split(topics):
        print('Training on Fold {} ---------'.format(fold))
        if model_type == 'hatformer':
            config, model, tokenizer = get_hatformer_model(extra_encoder_configs, model_path=model_path)
        elif model_type == 'longformer':
            config, model, tokenizer = get_longformer_model(extra_encoder_configs, model_path=model_path)
        else:
            print('No model type identified..')
            return

        wandb.init(settings=wandb.Settings(start_method="fork"), project="exp-quality-project", entity="milad-it", name='{}'.format(wandb_run_name))
        
        train_topics = topics[fold[0]]
        test_topics = topics[fold[1]]
        train_topics, valid_topics = train_test_split(train_topics, test_size=0.2, shuffle=True, random_state=123)
        
        train_df = df[df.topic.isin(train_topics)]
        valid_df = df[df.topic.isin(valid_topics)]
        test_df  = df[df.topic.isin(test_topics)]

        #balance the data
        train_df, y = ros.fit_resample(train_df, train_df['labels'])
        train_df['labels'] = y

        training_ds = Dataset.from_pandas(train_df)
        valid_ds    = Dataset.from_pandas(valid_df)
        test_ds     = Dataset.from_pandas(test_df)

        if model_type == 'hatformer':
            training_ds = training_ds.map(lambda examples: preprocess_hat_function(tokenizer, examples, max_seq_length, False), 
                                batched=True, load_from_cache_file=False, remove_columns=['labels'])
            valid_ds    = valid_ds.map(lambda examples: preprocess_hat_function(tokenizer, examples, max_seq_length, False), 
                                batched=True, load_from_cache_file=False, remove_columns=['labels'])
            test_ds     = test_ds.map(lambda examples: preprocess_hat_function(tokenizer, examples, max_seq_length, False), 
                                batched=True, load_from_cache_file=False, remove_columns=['labels'])
        else:
            training_ds = Dataset.from_dict(preprocess_function(tokenizer, train_df, input_clm=input_clm))
            valid_ds = Dataset.from_dict(preprocess_function(tokenizer, valid_df, input_clm=input_clm))
            test_ds = Dataset.from_dict(preprocess_function(tokenizer, test_df, input_clm=input_clm))

        print('Training {}, Valid {}, and Test {}'.format(len(training_ds), len(valid_ds), len(test_ds)))

        args = TrainingArguments(
                output_dir= '{}/{}/{}-fold'.format(output_dir, wandb_run_name, fold_idx),
                overwrite_output_dir=True,
                evaluation_strategy = "steps",
                save_strategy = "steps",
                logging_strategy="steps",
                save_total_limit=5,
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_train_epochs,
                weight_decay=0.01,
                eval_steps=eval_steps,
                logging_steps=eval_steps,
                load_best_model_at_end=True,
                metric_for_best_model='rmse'
            )


        trainer = Trainer(
            model,
            args,
            train_dataset=training_ds,
            eval_dataset=valid_ds,
            compute_metrics=lambda x: compute_metrics(x),
            tokenizer=tokenizer
        )

        if only_eval == False:
            trainer.train()

        eval_results = trainer.evaluate(test_ds)
        rmse_scores.append(eval_results['eval_rmse'])
        
        if only_eval == False: #only save results when we are training also
            model.save_pretrained('{}/{}/{}-fold'.format(output_dir, wandb_run_name, fold_idx))
            test_ds.to_json('{}/{}/{}-fold/test_set.json'.format(output_dir, wandb_run_name, fold_idx))
            json.dump(eval_results, open('{}/{}/{}-fold/eval_results.json'.format(output_dir, wandb_run_name, fold_idx), 'w'))
        
        fold_idx=fold_idx+1
        wandb.finish()
        
    return rmse_scores

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


if __name__=="__main__":
    
    
    parser = argparse.ArgumentParser(description='Train Models for Quality Prediction.')
    parser.add_argument('model_type', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('run_name', type=str)
    parser.add_argument('--feats_clms', type=str, default=None)
    parser.add_argument('--input_clm', type=str, required=False, default="turn_text")
    parser.add_argument('--n_splits', type=int, required=False, default=-1)
    parser.add_argument('--only_eval', action='store_true', default=False)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--eval_on_test', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    
    
    args = parser.parse_args()
    
    eli5_annotation_training_df = load_and_prepare_df('../../data/eli5_ds/annotation-results/MACE-measure/final_mace_predictions_training.pkl', '../../data/eli5_ds/annotation-results/MACE-measure/final_mace_rating_predictions.csv')
    
    eli5_annotation_testing_df = load_and_prepare_df('../../data/eli5_ds/annotation-results/MACE-measure/final_mace_predictions_testing.pkl', '../../data/eli5_ds/annotation-results/MACE-measure/final_mace_rating_predictions.csv')
    
    feats_clms = args.feats_clms.split(",") if args.feats_clms != None else []
    extra_encoders = [hat_former_feats_clms_encoders[x] for x in feats_clms] if args.model_type == 'hatformer' else [long_former_feats_clms_encoders[x] for x in feats_clms]
    
    #feats_clms=args.feats_clms
    #extra_encoders = [feats_clms_encoders[x] for x in feats_clms]

    print(feats_clms)
    print(extra_encoders)

    if args.n_splits == -1:
        
        if args.eval_on_test:
            eli5_annotation_testing_df = eli5_annotation_testing_df[['topic', 'input_texts', 'labels'] + feats_clms]
        else:
            print('Evaluating on the validation set')
            eli5_annotation_testing_df = None
            
        rmse_scores = train_and_evaluate_single_fold(args.model_type, args.output_path, 
                                    eli5_annotation_training_df[['topic', 'input_texts', 'labels'] + feats_clms], args.run_name,
                                    eli5_annotation_testing_df,
                                    input_clm='input_texts', num_train_epochs=args.num_epochs, eval_steps=50, lr=args.lr, batch_size=args.batch_size, 
                                    extra_encoder_configs=extra_encoders, only_eval=args.only_eval, model_path=args.model_path)
    else:
        rmse_scores = cross_fold_training_and_evaluation(args.model_type, args.output_path, 
                                    eli5_annotation_training_df[['topic', 'input_texts', 'labels'] + feats_clms], args.run_name, 
                                    input_clm='input_texts', num_train_epochs=10, eval_steps=50, lr=2e-5, batch_size=4, 
                                    extra_encoder_configs=extra_encoders, n_splits=args.n_splits, only_eval=args.only_eval, model_path=args.model_path)
        
    
    print(np.mean(rmse_scores), rmse_scores)