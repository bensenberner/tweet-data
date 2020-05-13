import transformers

from kaggle_data_loader import TweetDataset as NewTweetDataset
from old_loader import TweetDataset as OldTweetDataset
from test.test import DatasetTestCase
from utils import load_small_df

BERT_MODEL_TYPE = 'bert-base-cased'


class TestCompareOldDatasetToNewDataset(DatasetTestCase):
    def test(self):
        df = load_small_df()
        bert_tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE)
        old_dataset = OldTweetDataset(df, bert_tokenizer)
        new_dataset = NewTweetDataset(df, bert_tokenizer)
        for old_data, new_data in zip(old_dataset, new_dataset):
            self.assertDatasetItemEqual(old_data, new_data)
