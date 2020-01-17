import logging
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from logginghandler import TQDMLoggingHandler
from neural_nets.common.utility import evaluatation_scores
from neural_nets.models.transformers.global_args import TEMP_DIRECTORY, RESULT_FILE
from neural_nets.models.transformers.run_model import ClassificationModel
from project_config import SEED, ENGLISH_DATA_PATH

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

full = pd.read_csv(ENGLISH_DATA_PATH, sep='\t')

le = LabelEncoder()
full['label'] = le.fit_transform(full["subtask_a"])
full['text'] = full["tweet"]


full = full[['text', 'label']]
train, test = train_test_split(full, test_size=0.2, random_state=SEED)

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base',
                            use_cuda=False)  # You can set class weights by using the optional weight argument

# Train the model
logging.info("Started Training")
model.train_model(train)
logging.info("Finished Training")
# Evaluate the model
test_sentences = test['text'].tolist()
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
