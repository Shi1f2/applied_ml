# Hand-crafted regex features that look like classic email-spam tells.
# Used as auxiliary signals for the unsupervised Stage A spam filter.
import re

import pandas as pd


PATTERNS = {
    'subject_header': r'^subject\s*:',
    'url': r'https?://|www\.',
    'email_addr': r'\b[\w.-]+@[\w.-]+',
    'all_caps_word': r'\b[A-Z]{4,}\b',
    'money': r'\$|usd|gbp|\beur\b|\bdollar',
    'click_here': r'click\s+here|unsubscribe|free\s+\w+|act\s+now',
    'phone_like': r'\b\d{3,}[\s\-.]?\d{3,}[\s\-.]?\d{3,}\b',
}


def pattern_hits(text, pattern):
    return len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))


def signals_dataframe(texts):
    return pd.DataFrame({name: texts.apply(lambda t, p=pat: pattern_hits(t, p)) for name, pat in PATTERNS.items()})
