from multiprocessing import cpu_count

MODEL_TYPE = "bert"
MODEL_NAME = "bert-base-cased"
TEMP_DIRECTORY = "temp/data"
RESULT_FILE = "result.tsv"
INCLUDE_SUPPORT = True

global_args = {
    'output_dir': 'temp/outputs/',
    'cache_dir': 'temp/cache_dir/',

    'fp16': True,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 1,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 50,
    'save_steps': 2000,
    'save_model_every_epoch': True,
    'evaluate_during_training': False,
    'evaluate_during_training_steps': 50,
    'use_cached_eval_features': True,
    'save_eval_checkpoints': False,
    'tensorboard_dir': None,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,

    'wandb_project': {},
    'wandb_kwargs': {},
}

hasoc_args = {
    'output_dir': 'temp/outputs/hasoc',
    'cache_dir': 'temp/cache_dir/hasoc',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 3,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 50,
    'save_steps': 2000,
    'save_model_every_epoch': True,
    'evaluate_during_training': False,
    'evaluate_during_training_steps': 50,
    'use_cached_eval_features': True,
    'save_eval_checkpoints': False,
    'tensorboard_dir': None,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,

    'wandb_project': "HASOC_2020",
    'wandb_kwargs': {},
}