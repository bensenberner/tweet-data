## TODO:
- make threshold trainable
- do we need to use a DataLoader in prediction?
- try out using PyTorch Lightning for better code quality?
- update ls_find_start_end so that we don't need to weird min max index stuff inpred
- is Jaccard score for positive/negative low because it's including too many wrong tokens (high jaccard denom) or it isn't finding enough right tokens (low jaccard numerator)