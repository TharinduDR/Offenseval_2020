from transformers import BertTokenizer

from neural_nets.models.bert.model_config import ENGLISH_BERT_MODEL

tokenizer = BertTokenizer.from_pretrained(ENGLISH_BERT_MODEL)
max_input_length = tokenizer.max_model_input_sizes[ENGLISH_BERT_MODEL]


def get_tokenizer():
    return tokenizer


def get_maximum_length():
    return max_input_length


def tokenize_and_cut(sentence):
    tokens = get_tokenizer().tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens
