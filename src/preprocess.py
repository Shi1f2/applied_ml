# Shared text tokeniser used by Stage A, Stage B and the NLTK evaluation
# so train/val/test/external all see the same normalisation.
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Negation tokens are kept on purpose — dropping them flips sentiment polarity.
NEGATION_TOKENS = {'not', 'no', 'nor', 'never', 'none', "n't", 'cannot', 'cant', 'wont'}
_STOPWORDS = set(stopwords.words('english')) - NEGATION_TOKENS

_NON_TOKEN = re.compile(r"[^a-z']+")


def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = []
    for tok in tokens:
        # Preserve negations verbatim before any cleaning.
        if tok in NEGATION_TOKENS:
            cleaned.append(tok)
            continue
        tok = _NON_TOKEN.sub('', tok)
        if not tok or tok == "'":
            continue
        if tok in _STOPWORDS:
            continue
        # Drop stray single chars except 'i' and 'a' which are real words.
        if len(tok) == 1 and tok not in {'i', 'a'}:
            continue
        cleaned.append(tok)
    return cleaned


def tokenize_corpus(texts):
    return [tokenize(t) for t in texts]
