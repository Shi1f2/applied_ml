import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


NEGATION_TOKENS = {'not', 'no', 'nor', 'never', 'none', "n't", 'cannot', 'cant', 'wont'}
_STOPWORDS = set(stopwords.words('english')) - NEGATION_TOKENS

_NON_TOKEN = re.compile(r"[^a-z']+")


def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = []
    for tok in tokens:
        if tok in NEGATION_TOKENS:
            cleaned.append(tok)
            continue
        tok = _NON_TOKEN.sub('', tok)
        if not tok or tok == "'":
            continue
        if tok in _STOPWORDS:
            continue
        if len(tok) == 1 and tok not in {'i', 'a'}:
            continue
        cleaned.append(tok)
    return cleaned


def tokenize_corpus(texts):
    return [tokenize(t) for t in texts]
