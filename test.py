import unittest
from collections import namedtuple
from typing import List

import pandas as pd
import torch
import transformers
from torch import tensor, Tensor

from kaggle_data_loader import BertTokenLabels, TweetDataset

BERT_MODEL_TYPE = 'bert-base-cased'
Pandas = namedtuple('Pandas', ['Index', 'text', 'selected_text'])


class TestBertTokenLabels(unittest.TestCase):
    # TODO: rename this test when it becomes more OOP
    def test_create_bert_based_labels_for_row(self):
        bert_tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE)
        bert_token_labels = BertTokenLabels(bert_tokenizer)
        text = "hello worldd"
        selected_text = "worldd"
        row = Pandas(Index=0, text=text, selected_text=selected_text)
        self.assertEqual(["hello", "world", "##d"], bert_tokenizer.tokenize(text))

        # TODO: fix this end_idx thing being not an actual idx
        start_idx, end_idx, labels = bert_token_labels.create_bert_based_labels_for_row(row)

        self.assertEqual(1, start_idx)
        self.assertEqual(3, end_idx)  # exclusive end_idx
        self.assertListEqual([0, 1, 1], labels)

    def test_create_bert_based_labels_for_rows_no_errors(self):
        bert_tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE)
        bert_token_labels = BertTokenLabels(bert_tokenizer)

        df = pd.DataFrame({
            'text': ['hello worldd'],
            'selected_text': ['worldd']
        })
        result, error_indexes = bert_token_labels.create_bert_based_labels(df)

        expected_df_index = 0
        expected_bert_ids = [101, 19082, 1362, 1181, 102]  # I think this includes [CLS] and [SEQ]
        expected_start_idx = 1
        expected_end_idx_with_offset = 3
        expected_labels = [0, 1, 1]
        expected_result = [
            (
                expected_df_index,
                expected_bert_ids,
                expected_start_idx,
                expected_end_idx_with_offset,
                expected_labels
            )
        ]
        self.assertEqual(expected_result, result)
        self.assertEqual([], error_indexes)


class TestTweetDataset(unittest.TestCase):
    def assertDatasetItemEqual(self, expected_item: tuple, actual_item: tuple):
        for expected_element, actual_element in zip(expected_item, actual_item):
            if isinstance(expected_element, Tensor):
                self.assertTrue(torch.equal(expected_element, actual_element))
            else:
                self.assertEqual(expected_element, actual_element)

    def assertDatasetItemInList(self, expected_item: tuple, actual_list_of_items: List[tuple]):
        # TODO: this is kinda jank lol
        for actual_item in actual_list_of_items:
            try:
                self.assertDatasetItemEqual(expected_item, actual_item)
                return
            except (AssertionError, RuntimeError):
                pass
        self.fail("Unable to find item in list")

    def test(self):
        df = pd.DataFrame({
            'text': ['hello worldd', 'hi moon'],
            'selected_text': ['worldd', 'hi moon'],
            'sentiment': ['neutral', 'positive']
        })
        dataset = TweetDataset(df, transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE))
        expected_item_0 = (
            0,
            tensor([101, 19082, 1362, 1181, 102]),
            tensor([1, 1, 1, 1, 1]),
            tensor([2, 4]),
            tensor([0., 0., 1., 1., 0.]),
            tensor(2)
        )
        self.assertDatasetItemEqual(expected_item_0, dataset[0])
        self.assertDatasetItemInList(expected_item_0, list(dataset))
