import re

from logginghandler import TQDMLoggingHandler
import logging
import spacy

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])

nlp = spacy.load('el_core_news_md', disable=['parser', 'tagger', 'ner'])


def tokenizer(x):
    return [w.text.lower() for w in nlp(x) if not w.is_stop]


def clean_text(x):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
              '*', '+', '\\', '•', '~', '@', '£',
              '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
              '█', '½', 'à', '…',
              '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
              '¥', '▓', '—', '‹', '─',
              '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
              '¾', 'Ã', '⋅', '‘', '∞',
              '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
              '¹', '≤', '‡', '√', ]

    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')

    return x


def remove_names(x):
    for word in x.split():
        if word[0] == "@":
            x = x.replace(word, "")
    return x


def sep_digits(x):
    return " ".join(re.split('(\d+)', x))


def sep_punc(x):
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~؛،؟؛.»«”'
    out = []
    for char in x:
        if char in punc:
            out.append(' ' + char + ' ')
        else:
            out.append(char)
    return ''.join(out)


def replaceMultiple(main, replacements, new):
    for elem in replacements:
        if elem in main:
            main = main.replace(elem, new)

    return main


def normalize(x):
    x = x.replace('ά', 'α')
    x = x.replace('έ', 'ε')
    x = x.replace('ή', 'η')
    x = replaceMultiple(x, ['ί', 'ΐ', 'ϊ'], 'ι')
    x = x.replace('ό', 'ο')
    x = replaceMultiple(x, ['ύ', 'ΰ', 'ϋ'], 'υ')
    x = x.replace('ώ', 'ω')
    return x


def pipeline(x):
    x = remove_names(x)
    x = clean_text(x)
    x = sep_digits(x)
    x = normalize(x)
    return tokenizer(x)
