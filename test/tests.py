import unittest
from collections import namedtuple
from typing import List

import pandas as pd
import torch
import transformers
from torch import tensor, Tensor

from kaggle_data_loader import (
    LabelMaker,
    TokenizedText,
    TestTweetDataset,
    TrainTweetDataset,
    TrainData,
    TestData,
    Prediction,
    ModelPipeline,
)

BERT_MODEL_TYPE = "bert-base-cased"
Pandas = namedtuple("Pandas", ["Index", "text", "selected_text"])


class TweetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE)

    @staticmethod
    def _are_dataset_items_equal(expected_item: tuple, actual_item: tuple):
        for expected_element, actual_element in zip(expected_item, actual_item):
            if isinstance(expected_element, Tensor):
                if not torch.equal(expected_element, actual_element):
                    return False, expected_element, actual_element
            else:
                if not expected_element == actual_element:
                    return False, expected_element, actual_element
        return True, None, None

    def assertDatasetItemEqual(self, expected_item: tuple, actual_item: tuple):
        are_equal, expected, actual = self._are_dataset_items_equal(expected_item, actual_item)
        if not are_equal:
            self.fail(f"Expected {expected}, actual {actual}")

    def assertDatasetItemInList(self, expected_item: tuple, actual_list_of_items: List[tuple]):
        for actual_item in actual_list_of_items:
            are_equal, _, _ = self._are_dataset_items_equal(expected_item, actual_item)
            if are_equal:
                return
        self.fail("Unable to find item in list")


class TestTokenizedText(TweetTestCase):
    def test_properties(self):
        sample_text = "oh, dangg"
        tokenized_text = TokenizedText(self.bert_tokenizer, sample_text)
        self.assertListEqual(["oh", ",", "da", "##ng", "##g"], tokenized_text.bert_tokens)
        self.assertListEqual(["oh,", "dangg"], tokenized_text.spaced_orig_tokens)
        self.assertListEqual([0, 2], tokenized_text.spaced_orig_to_bert_idx)
        self.assertListEqual([0, 0, 1, 1, 1], tokenized_text.bert_to_spaced_orig_idx)

    def test_make_orig_substring_from_bert_idxes(self):
        sample_text = "oh, dangg"
        # for the benefit of the test reader
        self.assertEqual(
            ["oh", ",", "da", "##ng", "##g"], self.bert_tokenizer.tokenize(sample_text)
        )
        tokenized_text = TokenizedText(self.bert_tokenizer, sample_text)
        substring = tokenized_text.make_orig_substring_from_bert_idxes(0, 1)

        self.assertEqual("oh,", substring)


class TestLabelData(TweetTestCase):
    def test_wtf(self):
        # this test is to prove that huggingface's docs are NOT consistent. these are supposed to be the same
        text = "hello worldd"
        self.assertNotEqual(
            self.bert_tokenizer.encode(text),
            self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(text)),
        )

    def test_find_normal(self):
        text = "Sooo SAD I will miss"
        selected_text = "Sooo SAD"
        self.assertEqual(
            ["Soo", "##o", "SA", "##D", "I", "will", "miss"], self.bert_tokenizer.tokenize(text)
        )
        row = Pandas(Index=0, text=text, selected_text=selected_text)

        label = LabelMaker(self.bert_tokenizer).make(row)
        self.assertEqual([1, 1, 1, 1, 0, 0, 0], label)

    def test_handle_unk_token(self):
        text = "helllllllllllllllllllllllloooooooooooooooooooooooooooooooooooooooooooooooooooooooolllllllllllllllllllllllo worldd"
        selected_text = "worldd"
        row = Pandas(Index=0, text=text, selected_text=selected_text)
        self.assertEqual(["[UNK]", "world", "##d"], self.bert_tokenizer.tokenize(text))

        label = LabelMaker(self.bert_tokenizer).make(row)
        self.assertListEqual([0, 1, 1], label)

    def test_find_truncated_beginning(self):
        text = "Sooo SAD I will miss"
        selected_text = "oo SAD"
        row = Pandas(Index=0, text=text, selected_text=selected_text)

        label = LabelMaker(self.bert_tokenizer).make(row)

        self.assertEqual([1, 1, 1, 1, 0, 0, 0], label)

    def test_find_truncated_end_that_splits_into_clean_bert_tokens(self):
        text = "Sooo SAD I will miss"
        selected_text = "Sooo SA"
        # this truncation in real text removes exactly a single bert token
        self.assertEqual(
            ["Soo", "##o", "SA", "##D", "I", "will", "miss"], self.bert_tokenizer.tokenize(text)
        )
        self.assertEqual(["Soo", "##o", "SA"], self.bert_tokenizer.tokenize(selected_text))

        row = Pandas(Index=0, text=text, selected_text=selected_text)

        label = LabelMaker(self.bert_tokenizer).make(row)
        # notice this is DIFFERENT than usual!
        self.assertEqual([1, 1, 1, 0, 0, 0, 0], label)

    def test_find_can_handle_single_truncated_if_it_splits_it_into_a_bert_token(self):
        text = "Sooo SAD I will miss"
        selected_text = "SA"
        row = Pandas(Index=0, text=text, selected_text=selected_text)

        label = LabelMaker(self.bert_tokenizer).make(row)

        self.assertEqual([0, 0, 1, 0, 0, 0, 0], label)

    def test_find_truncated_both_ends(self):
        text = "Sooo SAD I will miss"
        selected_text = "AD I wil"
        row = Pandas(Index=0, text=text, selected_text=selected_text)
        with self.assertRaisesRegex(
            AssertionError, f"Could not find '{selected_text}' in '{text}'"
        ):
            LabelMaker(self.bert_tokenizer).make(row)

    def test_find_middle_malformed(self):
        text = "Sooo SAD I will miss"
        selected_text = "SAD M will"
        row = Pandas(Index=0, text=text, selected_text=selected_text)
        with self.assertRaisesRegex(
            AssertionError, f"Could not find '{selected_text}' in '{text}'"
        ):
            LabelMaker(self.bert_tokenizer).make(row)

    def test_cannot_find_random(self):
        text = "Sooo SAD I will miss"
        selected_text = "lemonade"
        row = Pandas(Index=0, text=text, selected_text=selected_text)
        with self.assertRaisesRegex(
            AssertionError, f"Could not find '{selected_text}' in '{text}'"
        ):
            LabelMaker(self.bert_tokenizer).make(row)

    def test_find_cannot_handle_single_truncated_into_non_matching_bert_token(self):
        text = "Sooo SAD I will miss"
        selected_text = "S"
        row = Pandas(Index=0, text=text, selected_text=selected_text)
        with self.assertRaisesRegex(
            AssertionError, f"Could not find '{selected_text}' in '{text}'"
        ):
            LabelMaker(self.bert_tokenizer).make(row)

    def test_find_cannot_go_off_beginning_a_bit(self):
        text = "Sooo SAD I will miss"
        selected_text = "oh Sooo"
        row = Pandas(Index=0, text=text, selected_text=selected_text)
        with self.assertRaisesRegex(
            AssertionError, f"Could not find '{selected_text}' in '{text}'"
        ):
            LabelMaker(self.bert_tokenizer).make(row)

    def test_find_cannot_go_off_the_end_a_bit(self):
        text = "Sooo SAD I will miss"
        selected_text = "miss you"
        row = Pandas(Index=0, text=text, selected_text=selected_text)
        with self.assertRaisesRegex(
            AssertionError, f"Could not find '{selected_text}' in '{text}'"
        ):
            LabelMaker(self.bert_tokenizer).make(row)

    def test_find_cannot_go_off_the_end_too_far(self):
        # this has been failing for a while. Maybe its okay. Up for discussion
        text = "Sooo SAD I will miss"
        selected_text = "miss you man"
        row = Pandas(Index=0, text=text, selected_text=selected_text)

        with self.assertRaisesRegex(
            AssertionError, f"Could not find '{selected_text}' in '{text}'"
        ):
            LabelMaker(self.bert_tokenizer).make(row)


class TestFindStartEnd(unittest.TestCase):
    def test_entire(self):
        raw_logits = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1])
        mask = torch.tensor([1, 1, 1, 1, 1])
        threshold = 0

        pred = ModelPipeline._find_max_subarray_idxes(raw_logits, mask, threshold)

        # note the max_logit_sum. That means that the last value was NOT included in the range.
        # this is because we assume the last token (with mask = 1) to be the [SEQ] token which we don't include
        expected_pred = Prediction(
            start_idx=0, inclusive_end_idx=3, max_logit_sum=torch.tensor(0.8)
        )
        self.assertEqual(expected_pred, pred)

    def test_last_is_masked(self):
        raw_logits = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1])
        mask = torch.tensor([1, 1, 1, 1, 0])
        threshold = 0

        pred = ModelPipeline._find_max_subarray_idxes(raw_logits, mask, threshold)

        expected_pred = Prediction(
            start_idx=0, inclusive_end_idx=2, max_logit_sum=torch.tensor(0.6)
        )
        self.assertEqual(expected_pred, pred)


class TestTestTweetDataset(TweetTestCase):
    def test_with_normal_data(self):
        df = pd.DataFrame(
            {
                "text": ["hello worldd", "hi moon"],
                "selected_text": ["worldd", "hi moon"],
                "sentiment": ["neutral", "positive"],
            }
        )
        dataset = TestTweetDataset(df, self.bert_tokenizer)
        expected_item_0 = TestData(
            idx=0,
            input_id_list=tensor([101, 19082, 1362, 1181, 102]),
            input_id_mask=tensor([1, 1, 1, 1, 1]),
            sentiment=tensor(2),
        )
        self.assertDatasetItemEqual(expected_item_0, dataset[0])
        self.assertDatasetItemInList(expected_item_0, list(dataset))


class TestTrainTweetDataset(TweetTestCase):
    def test_normal_data(self):
        df = pd.DataFrame(
            {
                "text": ["hello worldd", "hi moon"],
                "selected_text": ["worldd", "hi moon"],
                "sentiment": ["neutral", "positive"],
            }
        )
        dataset = TrainTweetDataset(df, self.bert_tokenizer)

        expected_item_0 = TrainData(
            idx=0,
            input_id_list=tensor([101, 19082, 1362, 1181, 102]),
            input_id_mask=tensor([1, 1, 1, 1, 1]),
            sentiment=tensor(2),
            input_id_is_selected=tensor([0.0, 0.0, 1.0, 1.0, 0.0]),
        )
        self.assertListEqual([], dataset.error_indexes)
        self.assertDatasetItemEqual(expected_item_0, dataset[0])
        self.assertDatasetItemInList(expected_item_0, list(dataset))

    def test_data_with_row_that_is_filtered_out(self):
        df = pd.DataFrame(
            {
                "text": ["bad bad dog", "selected text wrong", "hi cat"],
                "sentiment": ["negative", "neutral", "positive"],
                "selected_text": ["bad bad", "not present in text", "hi"],
            }
        )
        dataset = TrainTweetDataset(df, self.bert_tokenizer)

        # skip the middle df row of idx 1 because selected text was not present in text
        self.assertEqual([0, 2], dataset.indexes)
        self.assertTrue(
            tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]).equal(dataset.bert_attention_masks)
        )
        self.assertTrue(
            tensor([[101, 2213, 2213, 3676, 102], [101, 20844, 5855, 102, 0]]).equal(
                dataset.bert_input_id_lists
            )
        )
