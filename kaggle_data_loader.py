from collections import namedtuple
from typing import List, Tuple

import pandas as pd
import torch
import transformers
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from utils import RANDOM_SEED


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True


class TokenizedText:
    def __init__(self, bert_tokenizer, full_text):
        self.spaced_orig_tokens = full_text.split(" ")
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

    def __repr__(self):
        return str(self.bert_tokens)

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
        return " ".join(self.spaced_orig_tokens[orig_start_idx : orig_end_idx + 1])


class LabelData:
    def __init__(self, bert_tokenizer, row):
        """
        self.start_idx: the index within the bert-tokenized row.text in which the row.selected_text first appears
        self.end_idx: the index within the bert-tokenized row.text in which the row.selected_text last appears
            NOTE that these this range defined by these indexes might be a little too wide in order to make sure the full
            selected text was captured.
        """
        split_text = bert_tokenizer.tokenize(row.text)
        split_selected_text = bert_tokenizer.tokenize(row.selected_text)
        self.start_idx, self.end_idx = self.find(split_text, split_selected_text)
        if self.start_idx == -1:
            # TODO: get a count of how often this happens (in exploratory notebook)
            raise AssertionError(f"Could not find '{row.selected_text}' in '{row.text}'")
        self.label = [
            1 if self.start_idx <= idx < self.end_idx else 0 for idx in range(len(split_text))
        ]

    @staticmethod
    def find(haystack: List[str], needle: List[str]) -> Tuple[int, int]:
        """
        :param haystack: list in which `needle` is being looked for
        :param needle: we are trying to find the indexes in which `needle` is located in `haystack`
        :return: the start and end indexes that define where `needle` is located in `haystack`.
            If `needle` is not in `haystack` then we return (-1, -1)
        TODO: explain how we handle needles that are almost in haystack except they're truncated at the ends
        """
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
        return -1, -1


# namedtuple('TrainRow', ['#TODO: all the train row stuff including label'])


TestData = namedtuple("TestData", ["idxes", "all_bert_input_ids", "masks", "sentiments"])


# not meant to be directly instantiated in order to avoid confusion
class _TweetDataset(data.Dataset):
    SENTIMENT_MAP = {"negative": 0, "positive": 1, "neutral": 2}

    def __init__(self, df, bert_tokenizer):
        bert_input_ids_unpadded = []
        self.indexes = []  # the index of the row from the original df
        self.tokenized_strings = {}
        self.error_indexes = []
        # TODO: TODO: does this row_idx mess things up??
        for row in df.itertuples():
            try:
                self.indexes.append(row.Index)
                self.tokenized_strings[row.Index] = TokenizedText(bert_tokenizer, row.text)
                bert_input_ids_unpadded.append(bert_tokenizer.encode(row.text))
            except AssertionError:
                # TODO: is this the same index as it was before??
                # TODO: maybe do some sort of checking to indicate the error source
                self.error_indexes.append(row.Index)
        df_filtered = df.drop(self.error_indexes)
        # this is X, the input matrix we will feed into the model.
        self.all_bert_input_ids: torch.Tensor = pad_sequence(
            [torch.tensor(e) for e in bert_input_ids_unpadded], batch_first=True
        )
        self.bert_attention_masks = torch.min(self.all_bert_input_ids, torch.tensor(1)).detach()
        self.sentiments = torch.tensor(list(df_filtered["sentiment"].apply(self.SENTIMENT_MAP.get)))

    def __len__(self):
        return len(self.all_bert_input_ids)

    def __getitem__(self, idx):
        return TestData(
            self.indexes[idx],
            self.all_bert_input_ids[idx],
            self.bert_attention_masks[idx],
            self.sentiments[idx],
        )


# TODO: some sort of subclassing ?? Can namedtuples use inheritance??
TrainData = namedtuple(
    "TrainData",
    [
        "idxes",
        "all_bert_input_ids",
        "masks",
        "sentiments",
        "selected_ids_start_end_idx",
        "selected_text",
        "selected_ids",
    ],
)


class TrainTweetDataset(_TweetDataset):
    def __init__(self, df_with_selected_texts: pd.DataFrame, bert_tokenizer):
        super(TrainTweetDataset, self).__init__(df_with_selected_texts, bert_tokenizer)

        selected_ids_start_end_idx_raw = []
        # each label is a list of 1s and 0s, representing whether the corresponding bert token
        # was contained in the selected text or not
        labels: List[List[int]] = []
        # TODO: this is literally just a series from the original db...why am I doing this
        self.selected_text = []

        for row in df_with_selected_texts.itertuples():
            label_data = LabelData(bert_tokenizer, row)
            # TODO: TODO: couldn't I just get LabelData??
            selected_ids_start_end_idx_raw.append(
                # TODO: make sure that end_idx is always exclusive
                (label_data.start_idx, label_data.end_idx)
            )
            labels.append(label_data.label)
            self.selected_text.append(row.selected_text)
        # Append torch.tensor([0]) to start because input_tokens include [CLS] token as the first one
        # and [SEQ] as last one.
        self.selected_ids = pad_sequence(
            [torch.tensor([0] + label + [0]) for label in labels], batch_first=True
        ).float()  # Float for BCELoss
        # TODO^^ if we aren't using BCE loss do we still need this?

        # TODO: rename this somehow
        # Offset by one for [CLS]
        self.selected_ids_start_end = torch.tensor(
            [(start_idx + 1, end_idx + 1) for start_idx, end_idx in selected_ids_start_end_idx_raw]
        )

    def __getitem__(self, idx):
        return TrainData(
            self.indexes[idx],
            self.all_bert_input_ids[idx],
            self.bert_attention_masks[idx],
            self.sentiments[idx],
            self.selected_ids_start_end[idx],
            self.selected_text[idx],  # TODO: am I sure I need this?
            self.selected_ids[idx],
        )


class TestTweetDataset(_TweetDataset):
    pass


def create_train_test_datasets() -> Tuple[_TweetDataset]:
    pass


Prediction = namedtuple("Prediction", ["start_idx", "end_idx", "max_logit_sum"])


# TODO: test this and stuff
def ls_find_start_end(b, threshold):
    # TODO: TODO: prevent padding from being a possible idx
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

    return Prediction(start_idx=max_start_idx, end_idx=max_end_idx, max_logit_sum=max_so_far)


def differentiable_log_jaccard(logits, actual):
    A = torch.sigmoid(logits)
    B = actual
    C = A * B
    sum_c = torch.sum(C, axis=1)
    # Numerically stable log.
    return -torch.mean(
        torch.log(sum_c) - torch.log(torch.sum(A, axis=1) + torch.sum(B, axis=1) - sum_c)
    )


# Calculating jaccard from start_end indices
def jaccard_from_start_end(s1, e1, s2, e2):
    A_ = e1 - s1
    B_ = int(e2 - s2)
    C_ = int(max(min(e1, e2) - max(s1, s2), 0))
    return C_ / (A_ + B_ - C_)


def jaccard(str1, str2, debug=False):
    str1_set = set(str1.lower().split())
    str2_set = set(str2.lower().split())
    intersection = str1_set.intersection(str2_set)
    if debug:
        print(str1_set)
        print(str2_set)
        print(intersection)
    return float(len(intersection)) / (len(str1_set) + len(str2_set) - len(intersection))


class NetworkV3(nn.Module):
    """
    This network uses the last two hidden layers of the BERT transformer
    to develop predictions.
    """

    def __init__(self, bert_model_type):
        super().__init__()
        config = transformers.BertConfig.from_pretrained(bert_model_type)
        config.output_hidden_states = True
        self.bert = transformers.BertModel.from_pretrained(bert_model_type, config=config)

        # Freeze weights
        for param in self.bert.parameters():
            param.requires_grad = False

        self.d1 = nn.Dropout(0.1)

        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(config.hidden_size * 2, 1),
                nn.Linear(config.hidden_size * 2, 1),
                nn.Linear(config.hidden_size * 2, 1),
            ]
        )

    def forward(self, input_ids, masks, sentiment_labels) -> torch.Tensor:
        batch_size = input_ids.shape[0]

        last_hidden_state, _, all_hidden_states = self.bert(input_ids, masks)
        all_hidden_states = torch.cat((all_hidden_states[-1], all_hidden_states[-2]), dim=-1)

        all_hidden_states = self.d1(all_hidden_states)
        logits = torch.stack([l(all_hidden_states) for l in self.linear_layers], axis=1).view(
            batch_size, 3, -1
        )

        # Probably some gather magic to be done here.
        selected_id_logits = torch.stack(
            [logits[i][lbl] for i, lbl in enumerate(sentiment_labels)], axis=0
        )
        return selected_id_logits


class ModelPipeline:
    def __init__(
        self,
        dev,
        bert_tokenizer,
        model: nn.Module,
        learning_rate,
        selected_id_loss_fn,
        prediction_threshold=0.6,
    ):
        """
        # TODO: explain!
        :param dev:
        :param bert_tokenizer:
        :param model:
        :param learning_rate:
        :param selected_id_loss_fn:
        :param prediction_threshold:
        """
        self.bert_tokenizer = bert_tokenizer
        self.dev = dev
        self.model = model
        # TODO: allow optimizer to be specified?
        self.optim = Adam(model.parameters(), lr=learning_rate)
        self.selected_id_loss_fn = selected_id_loss_fn
        self.prediction_threshold = prediction_threshold

    def _get_loss(self, input_ids, masks, sentiment_labels, selected_ids) -> torch.Tensor:
        # TODO: docstring
        batch_size = input_ids.shape[0]
        selected_logits = self.model(
            input_ids.to(self.dev), masks.to(self.dev), sentiment_labels.to(self.dev),
        )
        loss_fn = self.selected_id_loss_fn
        return loss_fn(selected_logits.view(batch_size, -1), selected_ids)

    def _update_weights(self, loss) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def fit(self, train_data_loader: DataLoader, n_epochs):
        is_in_notebook = in_notebook()
        losses = []
        epoch_bar = tqdm(range(n_epochs)) if is_in_notebook else range(n_epochs)
        secondary_bar = tqdm(total=len(train_data_loader)) if is_in_notebook else None
        for epoch in epoch_bar:
            if is_in_notebook:
                secondary_bar.reset()
            # TODO: this is such a jank unpacking...
            for (_, input_ids, masks, sentiments, _, _, selected_ids) in train_data_loader:
                loss = self._get_loss(input_ids, masks, sentiments, selected_ids)
                losses.append(loss.item())
                self._update_weights(loss)
                if is_in_notebook:
                    secondary_bar.update(1)
            if not is_in_notebook:
                print(f"Epoch {epoch}")
        if is_in_notebook:
            secondary_bar.close()
        return losses

    # TODO: this should not be TrainTweetDataset
    def old_pred(self, dataset: _TweetDataset, batch_size=128):
        # TODO: docstring for threshold

        all_model_jaccard_scores = []
        all_benchmark_jaccard_scores = []
        # TODO: set these in params
        # TODO: do I even need data_loader in pred?
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for (
            df_idxes,
            input_ids,
            masks,
            sentiments,
            actual_start_end,
            selected_texts,
            _,
        ) in data_loader:
            with torch.no_grad():
                logits = self.model(
                    input_ids.to(self.dev), masks.to(self.dev), sentiments.to(self.dev),
                )
            prediction_vectors = torch.sigmoid(logits)
            pred_start_end = [
                ls_find_start_end(pvec, self.prediction_threshold) for pvec in prediction_vectors
            ]
            for (
                idx,
                (pred_start, pred_end, _),
                (actual_start, actual_end),
                actual_selected_text,
            ) in zip(df_idxes, pred_start_end, actual_start_end, selected_texts):
                # TODO: using tokenized_string is new!! is it working??
                tokenized_string = dataset.tokenized_strings[int(idx)]
                # TODO: update ls_find_start_end so that we don't need to do this
                pred_start = max(pred_start, 1)
                pred_end = min(pred_end, len(tokenized_string.bert_tokens) + 1)
                all_model_jaccard_scores.append(
                    (idx, jaccard_from_start_end(pred_start, pred_end, actual_start, actual_end))
                )
                # TODO(change into a explanatory note about undoing the [CLS] shift
                predicted_selected_text = tokenized_string.make_orig_substring_from_bert_idxes(
                    pred_start - 1, pred_end - 1
                )
                all_benchmark_jaccard_scores.append(
                    (idx, jaccard(actual_selected_text, predicted_selected_text))
                )
        return all_model_jaccard_scores, all_benchmark_jaccard_scores

    # TODO: change to TestTweetDataset (has no actual start or end idx)
    def pred_selected_text(self, dataset: _TweetDataset, batch_size=128):
        predicted_selected_texts = []
        # TODO: do I really need a dataloader for doing prediction?
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # TODO: don't explicitly unpack so this can handle train and test
        for data_row in data_loader:
            with torch.no_grad():
                logits = self.model(
                    data_row.all_bert_input_ids.to(self.dev),
                    data_row.masks.to(self.dev),
                    data_row.sentiments.to(self.dev),
                )
            prediction_vectors = torch.sigmoid(logits)
            preds = [
                ls_find_start_end(prediction_vector, self.prediction_threshold)
                for prediction_vector in prediction_vectors
            ]
            for idx, pred in zip(data_row.idxes, preds):
                tokenized_string = dataset.tokenized_strings[int(idx)]
                pred_start_idx = max(pred.start_idx, 1)
                pred_end_idx = min(pred.end_idx, len(tokenized_string.bert_tokens) + 1)
                # TODO(change into a explanatory note about undoing the [CLS] shift
                predicted_selected_text = tokenized_string.make_orig_substring_from_bert_idxes(
                    pred_start_idx - 1, pred_end_idx - 1
                )
                predicted_selected_texts.append(predicted_selected_text)
        return predicted_selected_texts

    def jaccard_scores(self, dataset, actual_selected_texts):
        predicted_selected_texts = self.pred_selected_text(dataset)
        # TODO: create a dataframe out of this?? something instead of using raw tuples
        return [
            (idx, jaccard(predicted_selected_text, actual_selected_text))
            for idx, (predicted_selected_text, actual_selected_text) in enumerate(
                zip(predicted_selected_texts, actual_selected_texts)
            )
        ]


def main():
    # TODO: I'm sometimes getting index out of bound errors?? uh oh!!1 (maybe should change the random seed around to repro)
    # TODO: need to have an example of a row that would raise an assertionerror and be filtered out
    torch.manual_seed(RANDOM_SEED)
    BERT_MODEL_TYPE = "bert-base-cased"
    bert_tokenizer_ = transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE)
    train_df_ = pd.DataFrame(
        {
            "text": ["hello worldd", "hi moon"],
            "sentiment": ["neutral", "positive"],
            "selected_text": ["worldd", "hi moon"],
        }
    )
    train_dataset = TrainTweetDataset(train_df_, bert_tokenizer_)
    train_data_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    dev = "cpu"
    lr = 0.001
    threshold = 0.6
    pipeline = ModelPipeline(
        dev=dev,
        bert_tokenizer=bert_tokenizer_,
        model=NetworkV3(BERT_MODEL_TYPE).to(dev),
        learning_rate=lr,
        selected_id_loss_fn=differentiable_log_jaccard,
        prediction_threshold=threshold,
    )
    pipeline.fit(train_data_loader, 1)
    # TODO: TODO: WHY DOES IT THROW LIST IDX OUT OF RANGE IF I RUN THIS TWICE??
    # pipeline.pred_selected_text(train_dataset)
    # actual_selected_texts = train_dataset.selected_text
    # print(pipeline.jaccard_scores(train_dataset, actual_selected_texts))

    test_df = pd.DataFrame(
        {"text": ["hi worldd", "hello moon"], "sentiment": ["neutral", "positive"]}
    )
    test_dataset = TestTweetDataset(test_df, bert_tokenizer_)
    print(pipeline.pred_selected_text(test_dataset))


if __name__ == "__main__":
    main()
