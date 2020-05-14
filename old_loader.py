from typing import List

import torch
import transformers
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils import data

from utils import load_small_df, RANDOM_SEED

BERT_MODEL_TYPE = "bert-base-cased"


class BenTokenizer:
    UNK = "[UNK]"

    def __init__(self, bert_tokenizer):
        self.bert_tokenizer = bert_tokenizer

    def tokenize_dep(self, string):
        space_split_strings = string.split(" ")
        bert_idx_to_original_char_idx = []
        bert_tokens = []
        curr_char_idx = 0
        bert_idx_to_original_tok_idx = []
        for (orig_token_idx, orig_token) in enumerate(space_split_strings):
            curr_bert_tokens = self.bert_tokenizer.tokenize(orig_token)
            for curr_bert_token in curr_bert_tokens:
                bert_idx_to_original_char_idx.append(curr_char_idx)
                bert_idx_to_original_tok_idx.append(orig_token_idx)
                if curr_bert_token.startswith("##"):
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
        spaced_tokens = original_str.split(" ")
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
        spaced_tokens = original_str.split(" ")
        orig_start_idx = tok_to_orig_index[start_idx]
        orig_end_idx = tok_to_orig_index[end_idx - 1]

        return " ".join(spaced_tokens[orig_start_idx : orig_end_idx + 1])

    @staticmethod
    def find(haystack: List[str], needle: List[str]):
        # returns the index in haystack that forms the beginning of the substring that matches needle.
        # returns -1 if needle is not in haystack.

        # TODO: collapse these loops
        for start_offset in [0, 1]:
            for end_offset in [0, -1]:
                truncated_needle = needle[0 + start_offset : len(needle) + end_offset]
                if len(truncated_needle) == 0:
                    continue
                for haystack_idx in range(len(haystack)):
                    # TODO: can I handle empty strings??
                    if (
                        haystack[haystack_idx] == truncated_needle[0]
                        and haystack[haystack_idx : haystack_idx + len(truncated_needle)]
                    ):
                        # always returning a range of len(needle)
                        return (
                            haystack_idx - start_offset,
                            haystack_idx - start_offset + len(needle),
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
            [1 if start_idx <= idx < end_idx else 0 for idx in range(len(split_text))],
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
                start_idx, end_idx, labels = self.create_bert_based_labels_for_row(
                    text, selected_text
                )
                bert_input_ids = self.bert_tokenizer.encode(text)
                index_labels.append((row_idx, bert_input_ids, start_idx, end_idx, labels))
            except AssertionError as e:
                error_indexes.append(row_idx)
        return (index_labels, error_indexes)


class TweetDataset(data.Dataset):
    SENTIMENT_MAP = {"negative": 0, "positive": 1, "neutral": 2}

    def __init__(self, df, bert_tokenizer):
        ben_tokenizer = BenTokenizer(bert_tokenizer)
        indexed_labels, error_indexes = ben_tokenizer.create_bert_based_labels_for_rows(
            df["text"], df["selected_text"]
        )
        df_filtered = df.drop(error_indexes)

        # For logging
        self.error_indexes = error_indexes
        self.indexes = [idx for idx, bert_input_ids, start_idx, end_idx, label in indexed_labels]
        self.bert_input_tokens = pad_sequence(
            [
                torch.tensor(bert_input_ids)
                for idx, bert_input_ids, start_idx, end_idx, label in indexed_labels
            ],
            batch_first=True,
        )
        self.bert_attention_mask = torch.min(self.bert_input_tokens, torch.tensor(1)).detach()

        # Append torch.tensor([0]) to start because input_tokens include [CLS] token as the first one, and [SEQ] as last one.
        self.selected_ids = pad_sequence(
            [
                torch.tensor([0] + label + [0])
                for idx, bert_input_ids, start_idx, end_idx, label in indexed_labels
            ],
            batch_first=True,
        ).float()  # Float for BCELoss

        # Offset by one for [CLS] and [SEQ]
        self.selected_ids_start_end = torch.tensor(
            [
                (start_idx + 1, end_idx + 1)
                for idx, bert_input_ids, start_idx, end_idx, label in indexed_labels
            ]
        )

        self.sentiment_labels = torch.tensor(
            [self.SENTIMENT_MAP[x] for x in df_filtered["sentiment"]]
        )

    def __len__(self):
        return len(self.bert_input_tokens)

    def __getitem__(self, idx):
        return (
            self.indexes[idx],
            self.bert_input_tokens[idx],
            self.bert_attention_mask[idx],
            self.selected_ids_start_end[idx],
            self.selected_ids[idx],
            self.sentiment_labels[idx],
        )


class NetworkV1(nn.Module):
    def __init__(self, bert_model_type, selected_id_loss_fn=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_model_type)

        # Freeze weights
        for param in self.bert.parameters():
            param.requires_grad = False

        config = transformers.BertConfig.from_pretrained(bert_model_type)
        self.selected_id_loss_fn = selected_id_loss_fn
        self.d1 = nn.Dropout(0.1)
        self.l1 = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, selected_ids, sentiment_labels):
        batch_size = input_ids.shape[0]

        last_hidden_state, _ = self.bert(input_ids, attention_mask)
        last_hidden_state = self.d1(last_hidden_state)
        logits = self.l1(last_hidden_state)

        loss_fn = self.selected_id_loss_fn
        loss = loss_fn(logits.view(batch_size, -1), selected_ids)

        return logits, loss


class NetworkV2(nn.Module):
    def __init__(self, bert_model_type, selected_id_loss_fn=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_model_type)

        # Freeze weights
        for param in self.bert.parameters():
            param.requires_grad = False

        config = transformers.BertConfig.from_pretrained(bert_model_type)
        self.selected_id_loss_fn = selected_id_loss_fn
        self.d1 = nn.Dropout(0.1)

        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, 1),
                nn.Linear(config.hidden_size, 1),
                nn.Linear(config.hidden_size, 1),
            ]
        )

    def forward(self, input_ids, attention_mask, selected_ids, sentiment_labels):
        batch_size = input_ids.shape[0]

        last_hidden_state, _ = self.bert(input_ids, attention_mask)
        last_hidden_state = self.d1(last_hidden_state)
        logits = torch.stack([l(last_hidden_state) for l in self.linear_layers], axis=1).view(
            batch_size, 3, -1
        )

        # Probably some gather magic to be done here.
        selected_logits = torch.stack(
            [logits[i][lbl] for i, lbl in enumerate(sentiment_labels)], axis=0
        )

        loss_fn = self.selected_id_loss_fn
        loss = loss_fn(selected_logits.view(batch_size, -1), selected_ids)
        return selected_logits, loss


class NetworkV3(nn.Module):
    """
    This network uses the last two hidden layers of the BERT transformer
    to develop predictions.
    """

    def __init__(self, bert_model_type, selected_id_loss_fn=nn.BCEWithLogitsLoss()):
        super().__init__()
        config = transformers.BertConfig.from_pretrained(bert_model_type)
        config.output_hidden_states = True
        self.bert = transformers.BertModel.from_pretrained(bert_model_type, config=config)

        # Freeze weights
        for param in self.bert.parameters():
            param.requires_grad = False

        self.selected_id_loss_fn = selected_id_loss_fn
        self.d1 = nn.Dropout(0.1)

        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(config.hidden_size * 2, 1),
                nn.Linear(config.hidden_size * 2, 1),
                nn.Linear(config.hidden_size * 2, 1),
            ]
        )

    def forward(self, input_ids, attention_mask, selected_ids, sentiment_labels):
        batch_size = input_ids.shape[0]

        last_hidden_state, _, all_hidden_states = self.bert(input_ids, attention_mask)
        all_hidden_states = torch.cat((all_hidden_states[-1], all_hidden_states[-2]), dim=-1)

        all_hidden_states = self.d1(all_hidden_states)
        logits = torch.stack([l(all_hidden_states) for l in self.linear_layers], axis=1).view(
            batch_size, 3, -1
        )

        # Probably some gather magic to be done here.
        selected_logits = torch.stack(
            [logits[i][lbl] for i, lbl in enumerate(sentiment_labels)], axis=0
        )

        loss_fn = self.selected_id_loss_fn
        loss = loss_fn(selected_logits.view(batch_size, -1), selected_ids)
        return selected_logits, loss


# Just for fun
def differentiable_log_jaccard(logits, actual):
    A = torch.sigmoid(logits)
    B = actual
    C = A * B
    sum_c = torch.sum(C, axis=1)
    # Numerically stable log.
    return -torch.mean(
        torch.log(sum_c) - torch.log(torch.sum(A, axis=1) + torch.sum(B, axis=1) - sum_c)
    )


def differentiable_log_jaccard_batch(logits, actual):
    A = torch.sigmoid(logits)
    B = actual
    C = A * B
    sum_c = torch.sum(C)
    return -torch.log(sum_c / (torch.sum(A) + torch.sum(B) - sum_c))


def ls_find_start_end(b, threshold=0.6):
    a = b - threshold
    max_so_far = a[0]
    cur_max = a[0]
    start_idx = 0
    max_start_idx, max_end_idx = 0, 0
    for i in range(1, len(a)):
        # cur_max = max(a[i], )
        # cur_max = a[i] + max(0, cur_max)
        if a[i] > cur_max + a[i]:
            cur_max = a[i]
            start_idx = i
        else:
            cur_max = cur_max + a[i]

        # max_so_far = max(max_so_far, cur_max)
        if max_so_far < cur_max:
            max_so_far = cur_max
            max_start_idx = start_idx
            max_end_idx = i + 1

    return max_start_idx, max_end_idx, max_so_far


# Calculating jaccard from start_end indices
def jaccard_from_start_end(s1, e1, s2, e2):
    A_ = e1 - s1
    B_ = int(e2 - s2)
    C_ = int(max(min(e1, e2) - max(s1, s2), 0))
    return C_ / (A_ + B_ - C_)


def jaccard(str1, str2, debug=False):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if debug:
        print(a)
        print(b)
        print(c)
    return float(len(c)) / (len(a) + len(b) - len(c))


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    train_df = load_small_df()
    bert_tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE)
    ben_tokenizer = BenTokenizer(bert_tokenizer)
    train_dataset = TweetDataset(train_df, bert_tokenizer)
    error_indexes = train_dataset.error_indexes
    train_df_filtered = train_df.drop(error_indexes)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    # TODO(pycharm): change
    n_epochs = 1
    lr = 0.001
    model = NetworkV3(BERT_MODEL_TYPE, differentiable_log_jaccard).to(dev)
    optim = Adam(model.parameters(), lr=lr)
    losses = []
    train_generator = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    # TODO(pycharm): couldnt get tqdm
    epoch_bar = range(n_epochs)
    secondary_bar = range(len(train_generator))
    for epoch in epoch_bar:
        for (
            df_idx,
            input_tokens,
            attention_mask,
            _,
            selected_ids,
            sentiment_labels,
        ) in train_generator:
            logits, loss = model(
                input_tokens.to(dev),
                attention_mask.to(dev),
                selected_ids.to(dev),
                sentiment_labels.to(dev),
            )
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {epoch}")

    threshold = 0.5
    train_df_map = {
        i: data
        for i, data in list(
            train_df_filtered.apply(
                lambda row: (
                    row.name,
                    (row.text, row.selected_text, *ben_tokenizer.tokenize_with_index(row.text)),
                ),
                1,
            )
        )
    }
    all_model_jaccard_scores = []
    all_benchmark_jaccard_scores = []
    for (
        df_idx,
        input_tokens,
        attention_mask,
        start_end,
        selected_ids,
        sentiment_labels,
    ) in train_generator:
        with torch.no_grad():
            logits, loss = model(
                input_tokens.to(dev),
                attention_mask.to(dev),
                selected_ids.to(dev),
                sentiment_labels.to(dev),
            )
        pvecs = torch.sigmoid(logits)
        pred_start_end = [ls_find_start_end(pvec, threshold) for pvec in pvecs]
        for i, (s1, e1, _), (s2, e2) in zip(df_idx, pred_start_end, start_end):
            (
                original_text,
                selected_text,
                bert_tokens,
                tok_to_orig_index,
                orig_to_tok_index,
            ) = train_df_map[int(i)]
            s1 = max(s1, 1)
            e1 = min(e1, len(bert_tokens) + 1)
            all_model_jaccard_scores.append((i, jaccard_from_start_end(s1, e1, s2, e2)))
            predicted_selected_text = BenTokenizer.recreate_original(
                original_text, tok_to_orig_index, s1 - 1, e1 - 1
            )
            all_benchmark_jaccard_scores.append(
                (i, jaccard(selected_text, predicted_selected_text))
            )
