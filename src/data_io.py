import pandas as pd

from . import paths


def load_train():
    return pd.read_csv(paths.TASK1_TRAIN)


def load_val():
    return pd.read_csv(paths.TASK1_VAL)


def load_test():
    return pd.read_csv(paths.TASK1_TEST)


def load_all():
    return load_train(), load_val(), load_test()
