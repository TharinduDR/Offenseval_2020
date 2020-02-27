import logging
import os
import shutil

import sklearn
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from algo.neural_nets.common.preprocessing.arabic_preprocessing import transformer_pipeline
from algo.neural_nets.common.utility import evaluatation_scores
from algo.neural_nets.models.transformers.args.arabic_args import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, \
    RESULT_FILE, arabic_args, SUBMISSION_FILE, DEV_RESULT_FILE, SUBMISSION_FOLDER
from algo.neural_nets.models.transformers.common.run_model import ClassificationModel
from project_config import SEED, ARABIC_TRAINING_PATH, ARABIC_DEV_PATH, ARABIC_TEST_PATH
from util.logginghandler import TQDMLoggingHandler
from util.reader import read_test_tsv, read_train_tsv

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

train = read_train_tsv(ARABIC_TRAINING_PATH)
dev = read_train_tsv(ARABIC_DEV_PATH)
test = read_test_tsv(ARABIC_TEST_PATH)

logging.info("Test size {}".format(str(test.shape[0])))

le = LabelEncoder()

train['label'] = le.fit_transform(train["subtask_a"])
train['text'] = train["tweet"]
train = train[['text', 'label']]
train['text'] = train['text'].apply(lambda x: transformer_pipeline(x))

dev['label'] = le.fit_transform(dev["subtask_a"])
dev['text'] = dev["tweet"]
dev = dev[['text', 'label']]
dev['text'] = dev['text'].apply(lambda x: transformer_pipeline(x))

test['text'] = test["tweet"]
test['text'] = test['text'].apply(lambda x: transformer_pipeline(x))

model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=arabic_args,
                            use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument

# Train the model
logging.info("Started Training")

if arabic_args["evaluate_during_training"]:
    train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
    model.train_model(train, eval_df=eval_df)

else:
    model.train_model(train, f1=sklearn.metrics.f1_score, accuracy=sklearn.metrics.accuracy_score)

logging.info("Finished Training")
# Evaluate the model

logging.info("Started Evaluation")
dev_sentences = dev['text'].tolist()

if arabic_args["evaluate_during_training"]:
    model = ClassificationModel(MODEL_TYPE, arabic_args["best_model_dir"], args=arabic_args,
                                use_cuda=torch.cuda.is_available())

dev_predictions, raw_outputs = model.predict(dev_sentences)

dev['predictions'] = dev_predictions

(tn, fp, fn, tp), accuracy, weighted_f1, macro_f1, weighted_recall, weighted_precision = evaluatation_scores(dev,
                                                                                                             'label',
                                                                                                             "predictions")

dev.to_csv(os.path.join(TEMP_DIRECTORY, DEV_RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

logging.info("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))
logging.info("Accuracy {}".format(accuracy))
logging.info("Weighted F1 {}".format(weighted_f1))
logging.info("Macro F1 {}".format(macro_f1))
logging.info("Weighted Recall {}".format(weighted_recall))
logging.info("Weighted Precision {}".format(weighted_precision))

logging.info("Finished Evaluation")

logging.info("Started Testing")
test_sentences = test['text'].tolist()

if arabic_args["evaluate_during_training"]:
    model = ClassificationModel(MODEL_TYPE, arabic_args["best_model_dir"], args=arabic_args,
                                use_cuda=torch.cuda.is_available())

test_predictions, raw_outputs = model.predict(test_sentences)

test['subtask_a'] = le.inverse_transform(test_predictions)
test = test[['id', 'subtask_a']]
test.to_csv(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER, RESULT_FILE), header=False, sep=',', index=False,
            encoding='utf-8')

shutil.make_archive(os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE), 'zip',
                    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

logging.info("Finished Testing")
