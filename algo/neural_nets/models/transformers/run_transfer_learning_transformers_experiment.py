import logging
import os

import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from algo.neural_nets.common.preprocessing.english_preprocessing import remove_words
from algo.neural_nets.common.utility import evaluatation_scores
from algo.neural_nets.models.transformers.args.english_args import MODEL_TYPE, MODEL_NAME
from algo.neural_nets.models.transformers.args.transfer_learning_args import TEMP_DIRECTORY, RESULT_FILE, \
    transfer_learning_args, CUT_OFF
from algo.neural_nets.models.transformers.common.run_model import ClassificationModel
from project_config import SEED, SUPPORT_ENGLISH_DATA_PATH
from util.filtering import filter_supportfile
from util.logginghandler import TQDMLoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def run_transfer_learning_experiment():
    if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

    full = pd.read_csv(SUPPORT_ENGLISH_DATA_PATH, sep='\t')
    full = filter_supportfile(full, CUT_OFF)

    le = LabelEncoder()
    train, test = train_test_split(full, test_size=0.2, random_state=SEED)

    train['label'] = le.fit_transform(train["subtask_a"])
    train = train[['text', 'label']]
    train['text'] = train['text'].apply(lambda x: remove_words(x))

    test['label'] = le.fit_transform(test["subtask_a"])
    test = test[['text', 'label']]
    test['text'] = test['text'].apply(lambda x: remove_words(x))

    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=transfer_learning_args,
                                use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument

    # Train the model
    logging.info("Started Training")

    if transfer_learning_args["evaluate_during_training"]:
        train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
        model.train_model(train, eval_df=eval_df)

    else:
        model.train_model(train, f1=sklearn.metrics.f1_score, accuracy=sklearn.metrics.accuracy_score)

    logging.info("Finished Training")
    # Evaluate the model
    test_sentences = test['text'].tolist()

    if transfer_learning_args["evaluate_during_training"]:
        model = ClassificationModel(MODEL_TYPE, transfer_learning_args["best_model_dir"], args=transfer_learning_args,
                                    use_cuda=torch.cuda.is_available())

    predictions, raw_outputs = model.predict(test_sentences)

    test['predictions'] = predictions

    (tn, fp, fn, tp), accuracy, weighted_f1, macro_f1, weighted_recall, weighted_precision = evaluatation_scores(test,
                                                                                                                 'label',
                                                                                                                 "predictions")

    test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

    logging.info("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))
    logging.info("Accuracy {}".format(accuracy))
    logging.info("Weighted F1 {}".format(weighted_f1))
    logging.info("Macro F1 {}".format(macro_f1))
    logging.info("Weighted Recall {}".format(weighted_recall))
    logging.info("Weighted Precision {}".format(weighted_precision))

    return transfer_learning_args['best_model_dir']
