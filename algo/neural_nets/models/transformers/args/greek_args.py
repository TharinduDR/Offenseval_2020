from multiprocessing import cpu_count

TEMP_DIRECTORY = "greek_temp/data"
TRAIN_FILE = "train.tsv"
TEST_FILE = "test.tsv"
DEV_RESULT_FILE = "dev_result.tsv"
SUBMISSION_FOLDER = "turkish_transformers"
SUBMISSION_FILE = "turkish_transformers"
RESULT_FILE = "results.csv"
MODEL_TYPE = "bert"
MODEL_NAME = "nlpaueb/bert-base-greek-uncased-v1"

greek_args = {
    'output_dir': 'greek_temp/outputs/',
    "best_model_dir": "greek_temp/outputs/best_model",
    'cache_dir': 'greek_temp/cache_dir/',

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
    'do_lower_case': True,

    'logging_steps': 50,
    'save_steps': 100,
    "no_cache": False,
    'save_model_every_epoch': True,
    'evaluate_during_training': True,
    'evaluate_during_training_steps': 100,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': True,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 100,
    "early_stopping_delta": 0,
}
