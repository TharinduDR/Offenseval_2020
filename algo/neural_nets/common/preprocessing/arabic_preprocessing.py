import re


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


def remove_words(x):
    x = x.replace('@USER', '')
    x = x.replace('URL', '')
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


def normalize(some_string):
    normdict = {
        'ة': 'ه',
        'أ': 'ا',
        'إ': 'ا',
        'ي': 'ى',

    }
    out = [normdict.get(x, x) for x in some_string]
    return ''.join(out)


def transformer_pipeline(x):
    x = remove_words(x)
    x = clean_text(x)
    x = normalize(x)

    return x
