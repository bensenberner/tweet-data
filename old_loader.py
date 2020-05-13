from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


class BenTokenizer:
    UNK = '[UNK]'

    def __init__(self, bert_tokenizer):
        self.bert_tokenizer = bert_tokenizer

    def tokenize_dep(self, string):
        space_split_strings = string.split(' ')
        bert_idx_to_original_char_idx = []
        bert_tokens = []
        curr_char_idx = 0
        bert_idx_to_original_tok_idx = []
        for (orig_token_idx, orig_token) in enumerate(space_split_strings):
            curr_bert_tokens = self.bert_tokenizer.tokenize(orig_token)
            for curr_bert_token in curr_bert_tokens:
                bert_idx_to_original_char_idx.append(curr_char_idx)
                bert_idx_to_original_tok_idx.append(orig_token_idx)
                if curr_bert_token.startswith('##'):
                    curr_char_idx += len(curr_bert_token) - 2
                elif curr_bert_token == self.UNK:
                    # entire current string is non-detokenizable
                    raise NotImplementedError("Cannot detokenize [UNK]")
                else:  # normal token
                    curr_char_idx += len(curr_bert_token)
                curr_char_idx += 1  # skip the space.
            bert_tokens.extend(curr_bert_tokens)
        return bert_tokens, bert_idx_to_original_char_idx, bert_idx_to_original_tok_idx

    def tokenize_with_index(self, original_str):
        spaced_tokens = original_str.split(' ')
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(spaced_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.bert_tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        return all_doc_tokens, tok_to_orig_index, orig_to_tok_index

    @staticmethod
    def recreate_original(original_str, tok_to_orig_index, start_idx, end_idx):
        spaced_tokens = original_str.split(' ')
        orig_start_idx = tok_to_orig_index[start_idx]
        orig_end_idx = tok_to_orig_index[end_idx - 1]

        return ' '.join(spaced_tokens[orig_start_idx: orig_end_idx + 1])

    @staticmethod
    def find(haystack: List[str], needle: List[str]):
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
        return (-1, -1)

    #     def create_bert_based_labels_for_row(self, text: str, selected_text: str):
    #         split_text = self.bert_tokenizer.tokenize(text)
    #         split_selected_text = self.bert_tokenizer.tokenize(selected_text)
    #         return split_text, split_selected_text

    def create_bert_based_labels_for_row(self, text, selected_text, is_debug=False):
        split_text = self.bert_tokenizer.tokenize(text)
        split_selected_text = self.bert_tokenizer.tokenize(selected_text)
        start_idx, end_idx = self.find(split_text, split_selected_text)
        if start_idx == -1:
            # TODO: print to stderr instead of stdout
            # TODO: get a count of how often this happens
            raise AssertionError(f"Could not find '{selected_text}' in '{text}'")
        if is_debug:
            print(split_text)
            print(split_selected_text)
        return (
            start_idx,
            end_idx,
            [1 if start_idx <= idx < end_idx else 0 for idx in range(len(split_text))]
        )

    def create_bert_based_labels_for_rows(self, texts, selected_texts):
        """
        # (first element in idx is tuple, second is label)
        # (list of idxes that have errors in them)
        """
        if texts.shape[0] != selected_texts.shape[0]:
            raise AssertionError("Mismatched number of rows")
        # TODO: how to do padding??
        index_labels = []
        error_indexes = []
        for row_idx, (text, selected_text) in enumerate(zip(texts, selected_texts)):
            try:
                start_idx, end_idx, labels = self.create_bert_based_labels_for_row(text, selected_text)
                bert_input_ids = self.bert_tokenizer.encode(text)
                index_labels.append((row_idx, bert_input_ids, start_idx, end_idx, labels))
            except AssertionError as e:
                error_indexes.append(row_idx)
        return (index_labels, error_indexes)


class TweetDataset(data.Dataset):
    SENTIMENT_MAP = {
        'negative': 0,
        'positive': 1,
        'neutral': 2
    }

    def __init__(self, df, bert_tokenizer):
        ben_tokenizer = BenTokenizer(bert_tokenizer)
        indexed_labels, error_indexes = ben_tokenizer.create_bert_based_labels_for_rows(df['text'], df['selected_text'])
        df_filtered = df.drop(error_indexes)

        # For logging
        self.error_indexes = error_indexes
        self.indexes = [idx for idx, bert_input_ids, start_idx, end_idx, label in indexed_labels]
        self.bert_input_tokens = pad_sequence(
            [torch.tensor(bert_input_ids) for idx, bert_input_ids, start_idx, end_idx, label in indexed_labels],
            batch_first=True
        )
        self.bert_attention_mask = torch.min(self.bert_input_tokens, torch.tensor(1)).detach()

        # Append torch.tensor([0]) to start because input_tokens include [CLS] token as the first one, and [SEQ] as last one.
        self.selected_ids = pad_sequence(
            [torch.tensor([0] + label + [0]) for idx, bert_input_ids, start_idx, end_idx, label in indexed_labels],
            batch_first=True
        ).float()  # Float for BCELoss

        # Offset by one for [CLS] and [SEQ]
        self.selected_ids_start_end = torch.tensor(
            [(start_idx + 1, end_idx + 1) for idx, bert_input_ids, start_idx, end_idx, label in indexed_labels])

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
