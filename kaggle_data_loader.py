from typing import List, Tuple, Iterable, Optional, Callable, NamedTuple

import pandas as pd
import torch
import transformers
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

RANDOM_SEED = 42


def in_notebook():
    try:
        from IPython import get_ipython

        instance = get_ipython()
        if not instance or "IPKernelApp" not in instance.config:  # pragma: no cover
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

    def make_orig_substring_from_bert_idxes(
        self, bert_start_idx: int, bert_inclusive_end_idx: int
    ) -> str:
        """
        Given a start and end index from the bert token list, this will create
        a substring from the original string composed of the bert tokens within
        that start and end bert index range.

        :param bert_start_idx: start index from the bert token list
        :param bert_inclusive_end_idx: end index from the bert token list
        """
        orig_start_idx = self.bert_to_spaced_orig_idx[bert_start_idx]
        orig_end_idx = self.bert_to_spaced_orig_idx[bert_inclusive_end_idx]
        return " ".join(self.spaced_orig_tokens[orig_start_idx : orig_end_idx + 1])


class LabelMaker:
    def __init__(self, bert_tokenizer):
        self.bert_tokenizer = bert_tokenizer

    def make(self, row) -> List[int]:
        text_bert_toks = self.bert_tokenizer.tokenize(row.text)
        sel_text_bert_toks = self.bert_tokenizer.tokenize(row.selected_text)
        bert_text_start_idx, bert_text_end_idx = self._find(text_bert_toks, sel_text_bert_toks)
        if bert_text_start_idx == -1:
            # this happens 372 times in version as of this commit for all of train.csv
            raise AssertionError(f"Could not find '{row.selected_text}' in '{row.text}'")
        return [
            1 if bert_text_start_idx <= idx <= bert_text_end_idx else 0
            for idx in range(len(text_bert_toks))
        ]

    @staticmethod
    def _is_needle_at_haystack_idx(haystack, hs_idx, needle):
        """
        Found this on stackoverflow. Takes advantage of short circuiting to avoid creating temporary slices
        :return: True if the needle is contained within haystack, where needle[0] is at haystack[hs_idx]
        """
        return haystack[hs_idx] == needle[0] and (
            haystack[hs_idx + 1 : hs_idx + len(needle)] == needle[1:]
        )

    def _find(self, haystack: List[str], needle: List[str]) -> Tuple[int, int]:
        """
        :param haystack: list in which `needle` is being looked for
        :param needle: we are trying to find the indexes in which `needle` is located in `haystack`
        :return: the start and end indexes that define where `needle` is located in `haystack`.
            If `needle` is not in `haystack` then we return (-1, -1)
        TODO: explain how we handle needles that are almost in haystack except they're truncated at the ends
        """
        # hs stands for haystack
        for start_offset in [0, 1]:
            for end_offset in [0, -1]:
                sub_needle = needle[0 + start_offset : len(needle) + end_offset]
                if len(sub_needle) == 0:
                    continue
                for hs_idx in range(len(haystack)):
                    if self._is_needle_at_haystack_idx(haystack, hs_idx, sub_needle):
                        return (
                            hs_idx - start_offset,
                            hs_idx - start_offset + len(needle) - 1,
                        )
        return -1, -1


class TestData(NamedTuple):
    idx: int
    input_id_list: torch.Tensor
    input_id_mask: torch.Tensor
    sentiment: torch.Tensor


# not meant to be directly instantiated in order to avoid confusion regarding subclassing
class _TweetDataset(data.Dataset):
    SENTIMENT_MAP = {"negative": 0, "positive": 1, "neutral": 2}

    def __init__(self, df, bert_tokenizer):
        self.indexes = []  # the index of the row from the original df
        bert_input_ids_unpadded = []
        self.tokenized_strings = {}
        sentiments_raw = []
        for row in df.itertuples():
            self.indexes.append(row.Index)
            bert_input_ids_unpadded.append(bert_tokenizer.encode(row.text, add_special_tokens=True))
            self.tokenized_strings[row.Index] = TokenizedText(bert_tokenizer, row.text)
            sentiments_raw.append(self.SENTIMENT_MAP.get(row.sentiment))
        # this is X, the input matrix we will feed into the model.
        self.bert_input_id_lists: torch.Tensor = pad_sequence(
            [torch.tensor(e) for e in bert_input_ids_unpadded], batch_first=True
        )
        self.bert_attention_masks = torch.min(self.bert_input_id_lists, torch.tensor(1)).detach()
        self.sentiments = torch.tensor(sentiments_raw)

    def __len__(self):
        return len(self.bert_input_id_lists)

    def __getitem__(self, idx):
        return TestData(
            idx=self.indexes[idx],
            input_id_list=self.bert_input_id_lists[idx],
            input_id_mask=self.bert_attention_masks[idx],
            sentiment=self.sentiments[idx],
        )


class TrainData(NamedTuple):
    idx: int
    input_id_list: torch.Tensor
    input_id_mask: torch.Tensor
    sentiment: torch.Tensor
    input_id_is_selected: torch.Tensor


class TrainTweetDataset(_TweetDataset):
    def __init__(self, df_with_selected_texts: pd.DataFrame, bert_tokenizer):
        # each label is a list of 1s and 0s, representing whether the corresponding bert token
        # was contained in the selected text or not
        labels: List[List[int]] = []
        self.error_indexes = []
        label_maker = LabelMaker(bert_tokenizer)
        for row in df_with_selected_texts.itertuples():
            try:
                labels.append(label_maker.make(row))
            except AssertionError:
                self.error_indexes.append(row.Index)
        df_filtered = df_with_selected_texts.drop(self.error_indexes)

        # initialize super with FILTERED df
        super(TrainTweetDataset, self).__init__(df_filtered, bert_tokenizer)

        # Append torch.tensor([0]) to start because input_tokens include [CLS] token as the first one
        # and [SEQ] as last one.
        self.selected_ids = pad_sequence(
            [torch.tensor([0] + label + [0]) for label in labels], batch_first=True
        ).float()  # Float for BCELoss

    def __getitem__(self, idx):
        return TrainData(
            idx=self.indexes[idx],
            input_id_list=self.bert_input_id_lists[idx],
            input_id_mask=self.bert_attention_masks[idx],
            sentiment=self.sentiments[idx],
            input_id_is_selected=self.selected_ids[idx],
        )


class TestTweetDataset(_TweetDataset):
    pass


class Prediction(NamedTuple):
    start_idx: int
    inclusive_end_idx: int
    max_logit_sum: torch.Tensor


def differentiable_log_jaccard(logits: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
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
        dev: str,
        bert_tokenizer,
        model: nn.Module,
        selected_id_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optim: Optional[Optimizer] = None,
        learning_rate: float = 0.001,
    ):
        """
        :param dev: where computations are to be run
        :param bert_tokenizer: duh
        :param model: just needs to implement forward()
        :param selected_id_loss_fn: takes in predicted logits and actual labels (which are between 0 and 1) for multiple examples, and returns the loss for each example
        :param learning_rate: learning rate for the optimizer
        """
        self.bert_tokenizer = bert_tokenizer
        self.dev = dev
        self.model = model
        self.optim = Adam(model.parameters(), lr=learning_rate) if not optim else optim
        self.selected_id_loss_fn = selected_id_loss_fn

    def _get_loss(self, train_data: TrainData) -> torch.Tensor:
        batch_size = train_data.input_id_list.shape[0]
        selected_logits = self.model(
            train_data.input_id_list.to(self.dev),
            train_data.input_id_mask.to(self.dev),
            train_data.sentiment.to(self.dev),
        )
        loss_fn = self.selected_id_loss_fn
        return loss_fn(
            selected_logits.view(batch_size, -1), train_data.input_id_is_selected.to(self.dev)
        )

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
            train_datas: Iterable[TrainData] = iter(train_data_loader)
            for train_data in train_datas:
                loss = self._get_loss(train_data)
                losses.append(loss.item())
                self._update_weights(loss)
                if is_in_notebook:
                    secondary_bar.update(1)
            if not is_in_notebook:
                print(f"Epoch {epoch}")
        if is_in_notebook:
            secondary_bar.close()
        return losses

    @staticmethod
    def ls_find_start_end(
        raw_logits: torch.Tensor, mask: torch.Tensor, prediction_threshold: float
    ):
        """
        :param raw_logits:
        :param mask:
        :param prediction_threshold: when, for a given input text, we attempt to predict the selected text
            we first produce a vector of logits representing the probability that each individual BERT token
            derived from the input text is or is not "selected."
        :return:
        """
        logits = raw_logits - prediction_threshold
        max_so_far = logits[0]
        cur_max = logits[0]
        start_idx = 0
        max_start_idx, max_end_idx = 0, 0
        num_tokens_including_cls_and_seq = int(mask.sum())
        for i in range(
            1, num_tokens_including_cls_and_seq - 1
        ):  # do this to exclude seq at the end
            if logits[i] > cur_max + logits[i]:
                cur_max = logits[i]
                start_idx = i
            else:
                cur_max = cur_max + logits[i]
            if max_so_far < cur_max:
                max_so_far = cur_max
                max_start_idx = start_idx
                max_end_idx = i

        return Prediction(
            start_idx=max_start_idx, inclusive_end_idx=max_end_idx, max_logit_sum=max_so_far
        )

    def pred_selected_text(
        self, dataset: _TweetDataset, prediction_threshold: float, batch_size=128
    ) -> List[str]:
        predicted_selected_texts = []
        # TODO: do I really need a dataloader for doing prediction?
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in data_loader:
            with torch.no_grad():
                logits = self.model(
                    batch.input_id_list.to(self.dev),
                    batch.input_id_mask.to(self.dev),
                    batch.sentiment.to(self.dev),
                )
            prediction_vectors = torch.sigmoid(logits)
            preds = [
                self.ls_find_start_end(prediction_vector, input_id_mask, prediction_threshold)
                for prediction_vector, input_id_mask in zip(prediction_vectors, batch.input_id_mask)
            ]
            for idx, pred in zip(batch.idx, preds):
                tokenized_string = dataset.tokenized_strings[int(idx)]
                pred_start_idx = max(pred.start_idx, 1)

                #  -1 offset since we're ignoring the [CLS] token at the beginning
                predicted_selected_text = tokenized_string.make_orig_substring_from_bert_idxes(
                    pred_start_idx - 1, pred.inclusive_end_idx - 1,
                )
                predicted_selected_texts.append(predicted_selected_text)
        return predicted_selected_texts

    def jaccard_scores(self, dataset, actual_selected_texts, prediction_threshold):
        predicted_selected_texts = self.pred_selected_text(dataset, prediction_threshold)
        return pd.Series(
            [
                jaccard(pred_sel_text, actual_sel_text)
                for pred_sel_text, actual_sel_text in zip(
                    predicted_selected_texts, actual_selected_texts
                )
            ]
        )


def main():
    torch.manual_seed(RANDOM_SEED)
    BERT_MODEL_TYPE = "bert-base-cased"
    bert_tokenizer_ = transformers.BertTokenizer.from_pretrained(BERT_MODEL_TYPE)
    train_df_ = pd.DataFrame(
        {
            "text": ["hello worldd", "selected text wrong", "hi moon"],
            "sentiment": ["neutral", "negative", "positive"],
            "selected_text": ["worldd", "not present in text", "hi moon"],
        }
    )
    train_dataset = TrainTweetDataset(train_df_, bert_tokenizer_)
    train_data_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    # TODO: make this trainable
    prediction_threshold = 0.6
    pipeline = ModelPipeline(
        dev=dev,
        bert_tokenizer=bert_tokenizer_,
        model=NetworkV3(BERT_MODEL_TYPE).to(dev),
        selected_id_loss_fn=differentiable_log_jaccard,
    )
    pipeline.fit(train_data_loader, 1)
    pipeline.pred_selected_text(train_dataset, prediction_threshold)
    actual_selected_texts = train_df_.loc[set(train_df_.index) - set(train_dataset.error_indexes)][
        "selected_text"
    ]
    print(pipeline.jaccard_scores(train_dataset, actual_selected_texts, prediction_threshold))

    test_df = pd.DataFrame(
        {"text": ["hi worldd", "hello moon"], "sentiment": ["neutral", "positive"]}
    )
    test_dataset = TestTweetDataset(test_df, bert_tokenizer_)
    print(pipeline.pred_selected_text(test_dataset, prediction_threshold))


if __name__ == "__main__":
    main()
