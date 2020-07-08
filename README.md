## TODO:
- make threshold trainable
- try out using PyTorch Lightning for better code quality?
- is Jaccard score for positive/negative low because it's including too many wrong tokens (high jaccard denom) or it isn't finding enough right tokens (low jaccard numerator)
- What is the precision/recall of each of the models for each of the sentiments?

A kaggle competition I participated in with @nh2liu
## Problem statement
[Kaggle competition](https://www.kaggle.com/c/tweet-sentiment-extraction)

A supervised problem in which we are given

Features:
- Text of a tweet (e.g. "running around makes me so happy, I love it.")
- Sentiment of the tweet (e.g. "positive")

and you the label is the **list of tokens within the tweet that supports the sentiment**. In this case, the tokens "so happy, I love it." support the "positivity" of the tweet "running around makes me so happy, I love it." The score of each prediction is the Jaccard index between the predicted support phrase and the actual support phrase (or "selected text" as the feature is named)

## Data Exploration
See TODO (link to python notebook)

## Framing the problem
There are a few ways to perform inference for this problem:
- Separately predict the start and end index of the selected text within the full tweet text. If start > end, predict the empty string.
- Autoregressively generate the selected text word-by-word based on the input text, almost like text summarization
- Predict the likelihood that an individual word from the original text was present in the selected text, and then come up with a way to construct the predicted selected text from those logits

The initial intuition for this problem was that it sounded like we could use an attention mechanism to determine which words were important. However, attention is typically an *intermediate* layer, so it wasn't clear which loss we would use or how we would "train" the attention directly. For example, let's say we attempted to predict the sentiment of the tweet: one of `{positive, neutral, negative}`. We could use cross-entropy loss. We could have an intermediate attention layer that would "highlight" which words for the input token were useful in predicting the sentiment, and we could use the activations of the attention layer to predict the selected text from the input.
If we pick the wrong words, how would we penalize the attention in this setup? It wasn't clear.

### Predicting likelihood of presence of a token in the selected text
Instead, we chose the last option, in which we directly attempt to predict which individual words from the original tweet are manifest in the selected text.
### Tokens, not words
Actually, we first chunked up the tweet into BERT tokens and attempted to predict which BERT tokens were present in the selected text, NOT which words were present. But for the purpose of this explanation I'll stick with talking about "words" as the units of a tweet. Let me know if that's confusing and I'll change it.

### Using a differentiable Jaccard index
The Jaccard index is non-differentiable. Instead, what we did is we converted the selected text into an *array of 1s and 0s* whose length is the number of words of the original tweet (let's call this length `n`). A `1` represented that this word was present in the selected text, whereas a `0` represented that it wasn't. Let's call this vector `A` (for `Actual`). Then, given a prediction vector `P`, of length `n`, which attempts to predict the likelihood of each word in the tweet being present in the selected text, the loss was:

$Loss = mean(\frac{I}{U}) = mean(log(I) - log(U))$

where `mean()` averages all the elements in a vector, `log()` performs an element-wise log of each element of the vector, and

$I = P \odot A$

and

$U = P + A - (P \odot A)$

where $\odot$ is element-wise multiplication. We use `log()` since dealing with subtraction is more numerically stable than dealing with division.

Normally, `I` and `U` represent sets derived from the set of predicted items and the set of actual items. For these equations, rather than dealing with sets, we are dealing with vectors that represent sets.

Here's some intuition for understanding how to represent the set concepts of "intersection" or "union" using vector concepts.

Imagine I have three objects: `X, Y, Z`. I have a set, $Q_{set}$, which contains `{X, Y}` and another set $R_{set}$ which contains `{X, Z}`. How might would I represent these sets as vectors?

Just as I defined the `A` vector above, I could define vectors such that elements in the vector represent the presence or absence of `X, Y, Z`. Let's say that element 0 is `X`, 1 is `Y`, and 2 is `Z`. Thus, $Q_{vec} = [1, 1, 0]$ and $R_{vec} = [1, 0, 1]$.

The intersection of $Q_{set}$ and $R_{set}$ (let's call it $I_{set}$) is `{X}`. That means that $I_{vec}$ is `[1, 0, 0]`. How might we have derived this from $Q_{vec}$ and $R_{vec}$?

Since we're only dealing with 1s and 0s here, the obvious answer would be to take the element-wise AND. But what about without using bit-arithmetic?

If we *multiply* $Q_{vec}$ and $R_{vec}$ together, we also get `[1, 0, 0]`. This feels similar how, in order to calculate the probability that two independent events occur, you *multiply* the probabilities together.

What about calculating the union, which is `{X, Y, Z}` or `[1, 1, 1]` (call it `U`)? Again, let's think about probabilities (maybe using a [Venn diagram](https://dr282zn36sxxg.cloudfront.net/datastreams/f-d%3Aa499b7d235ca3fade0f77b770bf4869fc84f7bb690ff64f2e01162cb%2BIMAGE_TINY%2BIMAGE_TINY.1)). `U` would be `Q + R - I`, since if you add `Q` and `R` you double count `I`. Does that work here?
 
```
 [1, 1, 0]
+[1, 0, 1]
-[1, 0, 0]

=[1, 1, 1]
```

Yes.

The same principle works with the probability vector $P$, which contains elements within the range $[0, 1]$. You could think of the elements of $P$ as representing the "partial presence" of elements in a set ("partially" choosing words from all the words in the original tweet to be in the selected text).

### Constructing the selected text

I've described how to predict the likelihood that a word from the original tweet is present in the selected text. But then how to actually construct the predicted selected text?
The logits might be "bumpy", e.g.

Tweet: `Delicious pancakes set my measly heart aflutter`

Actual selected text: `heart aflutter`

Selected text as binary label: `[0, 0, 0, 0, 0, 1, 1]`

Predicted binary label: `[0.75, 0.5, 0.01, 0.2, 0.22, 0.7, 0.9]`

There's two peaks of probability in this prediction: at the beginning and end of the sentence. Clearly, my pretend model thinks "delicious" is important for sentiment (which makes intuitive sense, but that's not what the label says. More on why I chose this particular label below). But "heart aflutter" also has high likelihood. We could do something like: pick out every token that has p>0.5 and stitch them together into selected text. Two problems with that:
- As mentioned above (TODO: can I link within this doc?), we are not actually predicting words, we are predicting BERT tokens. If we go with this approach, imagine if one word becomes three BERT tokens, and only two have p>0.5. What would we do, come up with a smart way to "finish" the word so that we don't include any partial words in the selected text? That sounds too complicated.
- If we include extraneous words, it will increase the `Union`, which will decrease the Jaccard index. I would rather focus on the "higher quality" group of tokens rather than any token with p>0.5

The reason why I chose this example is that, if you recall from the data exploration (TODO: make sure this is actually recallable), we discovered that *all* selected texts were continguous. So I want a method that only selects continguous subarrays from the original text.

What we settled on doing was finding the subarray of logits with the maximum value. The logits are all positive, first we subtract a constant threshold value from all the logits, and then we find the max subarray from those new values. How did we pick the threshold? We grid-searched. We found a different threshold for each sentiment, because we noticed that the neutral sentiment tweets tended to select almost all the tokens (although we also assumed that the neutral model would tend to give high likelihoods anyway).