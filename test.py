import unittest
from collections import namedtuple
from typing import List

import pandas as pd
import torch
import transformers
from torch import tensor, Tensor

from kaggle_data_loader import TweetDataset, LabelData

BERT_MODEL_TYPE = 'bert-base-cased'
Pandas = namedtuple('Pandas', ['Index', 'text', 'selected_text'])


class DatasetTestCase(unittest.TestCase):
    @staticmethod
    def _are_dataset_items_equal(expected_item: tuple, actual_item: tuple):
        for expected_element, actual_element in zip(expected_item, actual_item):
            if isinstance(expected_element, Tensor):
                if not torch.equal(expected_element, actual_element):
                    return False
            else:
                if not expected_element == actual_element:
                    return False
        return True

    def assertDatasetItemEqual(self, expected_item: tuple, actual_item: tuple):
        self.assertTrue(self._are_dataset_items_equal(expected_item, actual_item))

    def assertDatasetItemInList(self, expected_item: tuple, actual_list_of_items: List[tuple]):
        for actual_item in actual_list_of_items:
            if self._are_dataset_items_equal(expected_item, actual_item):
                return
        self.fail("Unable to find item in list")


class TestTweetDataset(DatasetTestCase):
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


class TestLabelData(unittest.TestCase):
    def test(self):
        bert_tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE)
        text = "hello worldd"
        selected_text = "worldd"
        row = Pandas(Index=0, text=text, selected_text=selected_text)
        self.assertEqual(["hello", "world", "##d"], bert_tokenizer.tokenize(text))

        # TODO: fix this end_idx thing being not an actual idx
        label = LabelData(bert_tokenizer, row)

        self.assertEqual(1, label.start_idx)
        self.assertEqual(3, label.end_idx)  # exclusive end_idx
        self.assertListEqual([0, 1, 1], label.label)
