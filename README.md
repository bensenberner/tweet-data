## TODO:
- update ls_find_start_end so that we don't need to weird min max index stuff inpred
- Ben investigates where that index out of bound flaky thing came
- can find() handle empty strings?
    - write a few test cases for find
- try out using PyTorch Lightning for better code quality?
- Why does pred_selected_text throw an error if I run it twice on the same data?
- change the end_idx everywhere to be an actual index not an exclusive index
- make threshold trainable