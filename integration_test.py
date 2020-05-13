import re
import unicodedata

import pandas as pd
import transformers

from kaggle_data_loader import TweetDataset as NewTweetDataset
from old_loader import TweetDataset as OldTweetDataset
from test import DatasetTestCase

BERT_MODEL_TYPE = 'bert-base-cased'


def load_small_df():
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    def clean(string):
        unicode_string = unicodedata.normalize('NFKD', string).replace('\xa0', ' ')
        return _RE_COMBINE_WHITESPACE.sub(' ', unicode_string).strip()

    train_df = pd.read_csv('train_small.csv')
    train_df['text'] = train_df['text'].astype(str).apply(clean)
    train_df['selected_text'] = train_df['selected_text'].astype(str).apply(clean)
    return train_df


class TestCompareOldDatasetToNewDataset(DatasetTestCase):
    def test(self):
        df = load_small_df()
        bert_tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE)
        old_dataset = OldTweetDataset(df, bert_tokenizer)
        new_dataset = NewTweetDataset(df, bert_tokenizer)
        for old_data, new_data in zip(old_dataset, new_dataset):
            self.assertDatasetItemEqual(old_data, new_data)
