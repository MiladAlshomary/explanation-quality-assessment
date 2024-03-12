
## Predicting Quality:

### Hatformer:

- CUDA_VISIBLE_DEVICES=0 python quality_prediction_experiment.py hatformer ../data/quality_models/hat-models/ hat-model-baseline --n_splits 5 &> hat-former-logs/hat-model-baseline.log
- CUDA_VISIBLE_DEVICES=2 python quality_prediction_experiment.py hatformer ../data/quality_models/hat-models/ hat-model-exp-act-encoder --n_splits 5 --feats_clms exp_act_label &> hat-former-logs/hat-model-exp-act-encoder.log
- CUDA_VISIBLE_DEVICES=0 python quality_prediction_experiment.py hatformer ../data/quality_models/hat-models/ hat-model-dlg-act-encoder --n_splits 5 --feats_clms dlg_act_label &> hat-former-logs/hat-model-dlg-act-encoder.log
- CUDA_VISIBLE_DEVICES=1 python quality_prediction_experiment.py hatformer ../data/quality_models/hat-models/ hat-model-topic-func-encoder --n_splits 5 --feats_clms topic_func_label &> hat-former-logs/hat-model-topic-func-encoder.log
- CUDA_VISIBLE_DEVICES=0  python quality_prediction_experiment.py hatformer ../data/quality_models/hat-models/ hat-model-all-encoder --n_splits 5 --feats_clms exp_act_label,dlg_act_label,topic_func_label &> hat-former-logs/hat-model-all-encoder.log

#### Single Fold:

- CUDA_VISIBLE_DEVICES=0 python quality_prediction_experiment.py hatformer ../data/quality_models/single-fold/hat-models/ single-fold-hat-model-baseline --lr 2e-6 --num_epochs 20
- CUDA_VISIBLE_DEVICES=1 python quality_prediction_experiment.py hatformer ../data/quality_models/single-fold/hat-models/ single-fold-hat-model-exp-act-encoder --lr 2e-6 --num_epochs 20 --feats_clms exp_act_label
- CUDA_VISIBLE_DEVICES=0 python quality_prediction_experiment.py hatformer ../data/quality_models/single-fold/hat-models/ single-fold-hat-model-exp-act-encoder-random --random_labels --lr 2e-6 --num_epochs 20 --feats_clms exp_act_label
- CUDA_VISIBLE_DEVICES=2 python quality_prediction_experiment.py hatformer ../data/quality_models/single-fold/hat-models/ single-fold-hat-model-dlg-act-encoder --lr 2e-6 --num_epochs 20 --feats_clms dlg_act_label
- CUDA_VISIBLE_DEVICES=1 python quality_prediction_experiment.py hatformer ../data/quality_models/single-fold/hat-models/ single-fold-hat-model-dlg-act-encoder-random --random_labels --lr 2e-6 --num_epochs 20 --feats_clms dlg_act_label
- CUDA_VISIBLE_DEVICES=0 python quality_prediction_experiment.py hatformer ../data/quality_models/single-fold/hat-models/ single-fold-hat-model-topic-func-encoder --lr 2e-6 --num_epochs 20 --feats_clms topic_func_label
- CUDA_VISIBLE_DEVICES=0 python quality_prediction_experiment.py hatformer ../data/quality_models/single-fold/hat-models/ single-fold-hat-model-topic-func-encoder-random --random_labels --lr 2e-6 --num_epochs 20 --feats_clms topic_func_label
- CUDA_VISIBLE_DEVICES=1 python quality_prediction_experiment.py hatformer ../data/quality_models/single-fold/hat-models/ hat-model-all-encoder --lr 2e-6 --num_epochs 20 --feats_clms exp_act_label,dlg_act_label,topic_func_label


### Longformer:

- CUDA_VISIBLE_DEVICES=0 python quality_prediction_experiment.py longformer ../data/quality_models/longformer-models/ longformer-model-baseline --n_splits 5 &> longformer-logs/longformer-model-baseline.log
- CUDA_VISIBLE_DEVICES=0 python quality_prediction_experiment.py longformer ../data/quality_models/longformer-models/ longformer-model-exp-act-encoder --n_splits 5 --feats_clms exp_act_label &> longformer-logs/longformer-model-exp-act-encoder.log
- CUDA_VISIBLE_DEVICES=0 python quality_prediction_experiment.py longformer ../data/quality_models/longformer-models/ longformer-model-dlg-act-encoder --n_splits 5 --feats_clms dlg_act_label &> longformer-logs/longformer-model-dlg-act-encoder.log
- CUDA_VISIBLE_DEVICES=1 python quality_prediction_experiment.py longformer ../data/quality_models/longformer-models/ longformer-model-topic-func-encoder --n_splits 5 --feats_clms topic_func_label &> longformer-logs/longformer-model-topic-func-encoder.log
- CUDA_VISIBLE_DEVICES=0  python quality_prediction_experiment.py longformer ../data/quality_models/longformer-models/ longformer-model-all-encoder --n_splits 5 --feats_clms exp_act_label,dlg_act_label,topic_func_label &> longformer-logs/longformer-model-all-encoder.log

#### Single Fold:
- CUDA_VISIBLE_DEVICES=1 python quality_prediction_experiment.py longformer ../data/quality_models/single-fold/longformer-models/ --lr 2e-5 --num_epochs 20 single-fold-longformer-model-baseline
- CUDA_VISIBLE_DEVICES=1 python quality_prediction_experiment.py longformer ../data/quality_models/single-fold/longformer-models/  --lr 2e-5 --num_epochs 20 single-fold-longformer-model-dlg-act-encoder --feats_clms dlg_act_label
- CUDA_VISIBLE_DEVICES=1 python quality_prediction_experiment.py longformer ../data/quality_models/single-fold/longformer-models/  --lr 2e-5 --num_epochs 20 single-fold-longformer-model-dlg-act-encoder-random --random_labels --feats_clms dlg_act_label
- CUDA_VISIBLE_DEVICES=0 python quality_prediction_experiment.py longformer ../data/quality_models/single-fold/longformer-models/ --lr 2e-5 --num_epochs 20 single-fold-longformer-model-exp-act-encoder --feats_clms exp_act_label
- CUDA_VISIBLE_DEVICES=1 python quality_prediction_experiment.py longformer ../data/quality_models/single-fold/longformer-models/ --lr 2e-5 --num_epochs 20 single-fold-longformer-model-exp-act-encoder-random --random_labels --feats_clms exp_act_label
- CUDA_VISIBLE_DEVICES=1 python quality_prediction_experiment.py longformer ../data/quality_models/single-fold/longformer-models/  --lr 2e-5 --num_epochs 20 single-fold-longformer-model-topic-func-encoder --feats_clms topic_func_label
- CUDA_VISIBLE_DEVICES=2 python quality_prediction_experiment.py longformer ../data/quality_models/single-fold/longformer-models/  --lr 2e-5 --num_epochs 20 single-fold-longformer-model-topic-func-encoder-random --random_labels --feats_clms topic_func_label
- CUDA_VISIBLE_DEVICES=1  python quality_prediction_experiment.py longformer ../data/quality_models/single-fold/longformer-models/  --lr 2e-5 --num_epochs 20 longformer-model-all-encoder --feats_clms exp_act_label,dlg_act_label,topic_func_label



## Predicting turn labels:

### Basic bert experiments:

- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert.py dlg_act_label ../data/bert/dlg_act_prediction/mixed_ds_models/model/ --model_name bert-base-uncased --ds all &> ./bert-logs/dlg_act_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert.py dlg_act_label ../data/bert/dlg_act_prediction/5lvls_ds_models/model/ --model_name bert-base-uncased --ds 5lvls &> ./bert-logs/dlg_act_label_5lvls_models.log
- CUDA_VISIBLE_DEVICES=2 python turn_label_prediction_experiment_with_bert.py dlg_act_label ../data/bert/dlg_act_prediction/eli5_ds_models/model/ --model_name bert-base-uncased --ds eli5 &> ./bert-logs/dlg_act_label_eli5_models.log

- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert.py exp_act_label ../data/bert/exp_act_prediction/mixed_ds_models/model/ --model_name bert-base-uncased --ds all &> ./bert-logs/exp_act_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert.py exp_act_label ../data/bert/exp_act_prediction/5lvls_ds_models/model/ --model_name bert-base-uncased --ds 5lvls &> ./bert-logs/exp_act_label_5lvls_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert.py exp_act_label ../data/bert/exp_act_prediction/eli5_ds_models/model/ --model_name bert-base-uncased --ds eli5 &> ./bert-logs/exp_act_label_eli5_models.log

- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert.py topic_func_label ../data/bert/topic_func_prediction/mixed_ds_models/model/ --model_name bert-base-uncased --ds all &> ./bert-logs/topic_func_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert.py topic_func_label ../data/bert/topic_func_prediction/5lvls_ds_models/model/ --model_name bert-base-uncased --ds 5lvls &> ./bert-logs/topic_func_label_5lvls_models.log
- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert.py topic_func_label ../data/bert/topic_func_prediction/eli5_ds_models/model/ --model_name bert-base-uncased --ds eli5 &> ./bert-logs/topic_func_label_eli5_models.log

OR

CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert.py ../data/turn-label-models/ --model_name bert-base-uncased &> ./bert-logs/all_logs.log


### Basic roberta experiments:

- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert.py dlg_act_label ../data/roberta/dlg_act_prediction/mixed_ds_models/model/ --model_name roberta-base --ds all &> ./roberta-logs/dlg_act_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert.py dlg_act_label ../data/roberta/dlg_act_prediction/5lvls_ds_models/model/ --model_name roberta-base --ds 5lvls &> ./roberta-logs/dlg_act_label_5lvls_models.log
- CUDA_VISIBLE_DEVICES=2 python turn_label_prediction_experiment_with_bert.py dlg_act_label ../data/roberta/dlg_act_prediction/eli5_ds_models/model/ --model_name roberta-base --ds eli5 &> ./roberta-logs/dlg_act_label_eli5_models.log

- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert.py exp_act_label ../data/roberta/exp_act_prediction/mixed_ds_models/model/ --model_name roberta-base --ds all &> ./roberta-logs/exp_act_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert.py exp_act_label ../data/roberta/exp_act_prediction/5lvls_ds_models/model/ --model_name roberta-base --ds 5lvls &> ./roberta-logs/exp_act_label_5lvls_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert.py exp_act_label ../data/roberta/exp_act_prediction/eli5_ds_models/model/ --model_name roberta-base --ds eli5 &> ./roberta-logs/exp_act_label_eli5_models.log

- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert.py topic_func_label ../data/roberta/topic_func_prediction/mixed_ds_models/model/ --model_name roberta-base --ds all &> ./roberta-logs/topic_func_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert.py topic_func_label ../data/roberta/topic_func_prediction/5lvls_ds_models/model/ --model_name roberta-base --ds 5lvls &> ./roberta-logs/topic_func_label_5lvls_models.log
- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert.py topic_func_label ../data/roberta/topic_func_prediction/eli5_ds_models/model/ --model_name roberta-base --ds eli5 &> ./roberta-logs/topic_func_label_eli5_models.log

OR


CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert.py ../data/turn-label-models/ --model_name roberta-base &> ./roberta-logs/all_logs.log

### BERT seq experiments:
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert_seq.py dlg_act_label ../data/bert_seq/dlg_act_prediction/mixed_ds_models/model/ 10 --model_name bert-base-uncased --ds all &> ./bert_seq_logs/dlg_act_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert_seq.py dlg_act_label ../data/bert_seq/dlg_act_prediction/5lvls_ds_models/model/ 10 --model_name bert-base-uncased --ds 5lvls &> ./bert_seq_logs/dlg_act_label_5lvls_models.log
- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert_seq.py dlg_act_label ../data/bert_seq/dlg_act_prediction/eli5_ds_models/model/ 10 --model_name bert-base-uncased --ds eli5 &> ./bert_seq_logs/dlg_act_label_eli5_models.log

- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert_seq.py exp_act_label ../data/bert_seq/exp_act_prediction/mixed_ds_models/model/ 10 --model_name bert-base-uncased --ds all &> ./bert_seq_logs/exp_act_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert_seq.py exp_act_label ../data/bert_seq/exp_act_prediction/eli5_ds_models/model/ 10 --model_name bert-base-uncased --ds eli5 &> ./bert_seq_logs/exp_act_label_eli5_models.log
- CUDA_VISIBLE_DEVICES=2 python turn_label_prediction_experiment_with_bert_seq.py exp_act_label ../data/bert_seq/exp_act_prediction/5lvls_ds_models/model/ 10 --model_name bert-base-uncased --ds 5lvls &> ./bert_seq_logs/exp_act_label_5lvls_models.log

- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert_seq.py topic_func_label ../data/bert_seq/topic_func_prediction/mixed_ds_models/model/ 4 --model_name bert-base-uncased --ds all &> ./bert_seq_logs/topic_func_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert_seq.py topic_func_label ../data/bert_seq/topic_func_prediction/5lvls_ds_models/model/ 4 --model_name bert-base-uncased --ds 5lvls &> ./bert_seq_logs/topic_func_label_5lvls_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert_seq.py topic_func_label ../data/bert_seq/topic_func_prediction/eli5_ds_models/model/ 4 --model_name bert-base-uncased --ds eli5 &> ./bert_seq_logs/topic_func_label_eli5_models.log

OR

CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert_seq.py ../data/turn-label-models/seq --model_name bert-base-uncased &> ./bert_seq_logs/all_logs.log

### RoBERTa seq experiments:
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert_seq.py dlg_act_label ../data/roberta_seq/dlg_act_prediction/mixed_ds_models/model/ 10 --model_name roberta-base --ds all &> ./roberta_seq_logs/dlg_act_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert_seq.py dlg_act_label ../data/roberta_seq/dlg_act_prediction/5lvls_ds_models/model/ 10 --model_name roberta-base --ds 5lvls &> ./roberta_seq_logs/dlg_act_label_5lvls_models.log
- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert_seq.py dlg_act_label ../data/roberta_seq/dlg_act_prediction/eli5_ds_models/model/ 10 --model_name roberta-base --ds eli5 &> ./roberta_seq_logs/dlg_act_label_eli5_models.log

- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert_seq.py exp_act_label ../data/roberta_seq/exp_act_prediction/mixed_ds_models/model/ 10 --model_name roberta-base --ds all &> ./roberta_seq_logs/exp_act_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert_seq.py exp_act_label ../data/roberta_seq/exp_act_prediction/eli5_ds_models/model/ 10 --model_name roberta-base --ds eli5 &> ./roberta_seq_logs/exp_act_label_eli5_models.log
- CUDA_VISIBLE_DEVICES=2 python turn_label_prediction_experiment_with_bert_seq.py exp_act_label ../data/roberta_seq/exp_act_prediction/5lvls_ds_models/model/ 10 --model_name roberta-base --ds 5lvls &> ./roberta_seq_logs/exp_act_label_5lvls_models.log

- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert_seq.py topic_func_label ../data/roberta_seq/topic_func_prediction/mixed_ds_models/model/ 4 --model_name roberta-base --ds all &> ./roberta_seq_logs/topic_func_label_mixed_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert_seq.py topic_func_label ../data/roberta_seq/topic_func_prediction/5lvls_ds_models/model/ 4 --model_name roberta-base --ds 5lvls &> ./roberta_seq_logs/topic_func_label_5lvls_models.log
- CUDA_VISIBLE_DEVICES=1 python turn_label_prediction_experiment_with_bert_seq.py topic_func_label ../data/roberta_seq/topic_func_prediction/eli5_ds_models/model/ 4 --model_name roberta-base --ds eli5 &> ./roberta_seq_logs/topic_func_label_eli5_models.log

OR


CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert_seq.py ../data/turn-label-models/seq --model_name roberta-base &> ./roberta_seq_logs/all_logs.log

### Training final turn-label prediction model:
- CUDA_VISIBLE_DEVICES=0 python turn_label_prediction_experiment_with_bert_seq.py exp_act_label ../data/bert_seq/exp_act_prediction_final/mixed_ds_models/model/ 10 --ds all --final &> ./bert_seq_logs/exp_act_label_mixed_models_final.log
