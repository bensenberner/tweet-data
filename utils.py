import re
import unicodedata

import pandas as pd

RANDOM_SEED = 42


def load_small_df():
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    def clean(string):
        unicode_string = unicodedata.normalize('NFKD', string).replace('\xa0', ' ')
        return _RE_COMBINE_WHITESPACE.sub(' ', unicode_string).strip()

    train_df = pd.read_csv('train_small.csv')
    train_df['text'] = train_df['text'].astype(str).apply(clean)
    train_df['selected_text'] = train_df['selected_text'].astype(str).apply(clean)
    return train_df
