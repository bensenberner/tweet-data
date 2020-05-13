from typing import List, Tuple

import pandas as pd
import transformers
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


class TokenizedString:
    def __init__(self, bert_tokenizer, full_text, selected_text):
        self.selected_text = selected_text
        self.spaced_orig_tokens = full_text.split(' ')
        # list of len(original_str.split(' ')). Each index represents an orig token,
        # each value is the index of the FIRST bert token that corresponds to the orig token
        self.bert_to_spaced_orig_idx = []

        # list of len(bert tokens). Each index corresponds to a bert token, each
        # value is the index in spaced_original_tokens from which the bert token was derived
        self.spaced_orig_to_bert_idx = []

        # the bert tokens derived from the original string
        self.bert_tokens = []
        for (spaced_orig_idx, spaced_orig_token) in enumerate(self.spaced_orig_tokens):
            self.spaced_orig_to_bert_idx.append(len(self.bert_tokens))
            sub_tokens = bert_tokenizer.tokenize(spaced_orig_token)
            for sub_token in sub_tokens:
                self.bert_to_spaced_orig_idx.append(spaced_orig_idx)
                self.bert_tokens.append(sub_token)

    def make_orig_substring_from_bert_idxes(self, bert_start_idx: int, bert_end_idx: int) -> str:
        """
        Given a start and end index from the bert token list, this will create
        a substring from the original string composed of the bert tokens within
        that start and end bert index range.

        :param bert_start_idx: start index from the bert token list
        :param bert_end_idx: end index from the bert token list
        """
        orig_start_idx = self.bert_to_spaced_orig_idx[bert_start_idx]
        orig_end_idx = self.bert_to_spaced_orig_idx[bert_end_idx - 1]
        return ' '.join(
            self.spaced_orig_tokens[orig_start_idx: orig_end_idx + 1]
        )


class TokenizedStrings:
    def __init__(self, bert_tokenizer, train_df: pd.DataFrame):
        self.data = {
            row_idx: TokenizedString(
                bert_tokenizer=bert_tokenizer,
                full_text=row.text,
                selected_text=row.selected_text
            )
            for row_idx, row in train_df.iterrows()
        }

    def __getitem__(self, idx):
        return self.data[idx]


# TODO: take out new
class NEWBertTokenLabel:
    def __init__(self, bert_tokenizer, row):
        split_text = bert_tokenizer.tokenize(row.text)
        split_selected_text = bert_tokenizer.tokenize(row.selected_text)
        self.start_idx, self.end_idx = self.find(split_text, split_selected_text)
        if self.start_idx == -1:
            # TODO: print to stderr instead of stdout
            # TODO: get a count of how often this happens
            raise AssertionError(f"Could not find '{row.selected_text}' in '{row.text}'")
        self.label = [1 if self.start_idx <= idx < self.end_idx else 0 for idx in range(len(split_text))]

    @staticmethod
    def find(haystack: List[str], needle: List[str]) -> Tuple[int, int]:
        # returns the index in haystack that forms the beginning of the substring that matches needle.
        # returns -1 if needle is not in haystack.

        # TODO: collapse these loops
        for start_offset in [0, 1]:
            for end_offset in [0, -1]:
                truncated_needle = needle[0 + start_offset: len(needle) + end_offset]
                if len(truncated_needle) == 0:
                    continue
                for haystack_idx in range(len(haystack)):
                    # TODO: can I handle empty strings??
                    if (
                        haystack[haystack_idx] == truncated_needle[0]
                        and haystack[haystack_idx:haystack_idx + len(truncated_needle)]
                    ):
                        # always returning a range of len(needle)
                        return (
                            haystack_idx - start_offset,
                            haystack_idx - start_offset + len(needle)
                        )
        return -1, -1


class TweetDataset(data.Dataset):
    SENTIMENT_MAP = {
        'negative': 0,
        'positive': 1,
        'neutral': 2
    }

    def __init__(self, df, bert_tokenizer):
        # TODO: simply use init instead of another thing
        # bert_token_labels = BertTokenLabels(bert_tokenizer)
        selected_ids_start_end_idx_raw = []
        # each label is a list of 1s and 0s, representing whether the corresponding bert token
        # was contained in the selected text or not
        labels: List[List[int]] = []
        bert_input_ids_unpadded = []
        self.indexes = []
        self.error_indexes = []
        # TODO: TODO: does this row_idx mess things up??
        for row in df.itertuples():
            try:
                # TODO: create new name for this
                new_label = NEWBertTokenLabel(bert_tokenizer, row)
                # start_idx, end_idx, label = bert_token_labels.create_label_list_for_row(row)
                self.indexes.append(row.Index)
                # TODO (speedup): you are already tokenizing in create_label_list. tokenize outside and pass it in
                bert_input_ids = bert_tokenizer.encode(row.text)
                bert_input_ids_unpadded.append(bert_input_ids)
                selected_ids_start_end_idx_raw.append((new_label.start_idx, new_label.end_idx))
                labels.append(new_label.label)
            except AssertionError:
                # TODO: is this the same index as it was before??
                # TODO: maybe do some sort of checking to indicate the error source
                self.error_indexes.append(row.Index)
        df_filtered = df.drop(self.error_indexes)
        # TODO: rename since "tokens" is overloaded. these are integer ids, not strings.
        self.bert_input_tokens = pad_sequence(
            [torch.tensor(e) for e in bert_input_ids_unpadded],
            batch_first=True
        )
        # TODO: rename to masks
        self.bert_attention_mask = torch.min(self.bert_input_tokens, torch.tensor(1)).detach()
        # TODO: rename this somehow

        # Offset by one for [CLS] and [SEQ]
        self.selected_ids_start_end = torch.tensor([
            (start_idx + 1, end_idx + 1)
            for start_idx, end_idx in selected_ids_start_end_idx_raw
        ])

        # Append torch.tensor([0]) to start because input_tokens include [CLS] token as the first one
        # and [SEQ] as last one.
        self.selected_ids = pad_sequence(
            [torch.tensor([0] + label + [0]) for label in labels],
            batch_first=True
        ).float()  # Float for BCELoss
        # TODO: use pd.apply or something
        self.sentiment_labels = torch.tensor([self.SENTIMENT_MAP[x] for x in df_filtered['sentiment']])

    def __len__(self):
        return len(self.bert_input_tokens)

    def __getitem__(self, idx):
        return (
            self.indexes[idx],
            self.bert_input_tokens[idx],
            self.bert_attention_mask[idx],
            self.selected_ids_start_end[idx],
            self.selected_ids[idx],
            self.sentiment_labels[idx]
        )


if __name__ == "__main__":
    # TODO: load in the actual original data, do the old way vs the new way, make sure no regression
    # (could do a sample of like bottom 5 rows to make it easier to compare the diff. then do all rows)
    original_string = "I love the recursecenter its so #coolCantBelieve"
    BERT_MODEL_TYPE = 'bert-base-cased'
    bert_tokenizer_ = transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE)
    df = pd.DataFrame({
        'text': ['hello worldd', 'hi moon'],
        'selected_text': ['worldd', 'hi moon'],
        'sentiment': ['neutral', 'positive']
    })
    dataset = TweetDataset(df, bert_tokenizer_)
    print(list(dataset))
