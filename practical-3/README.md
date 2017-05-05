# practical 3
problem description in `practical3.md`

## task 1: RNN text classification
### questions

1. benefits/downsides of RNN-based text classification over BOW?
* pro RNN
 * can take structure of the text into account
 * reads the text as a sequence of tokens, which is very close to how humans process it (instead of the rather arbitray count-based BOW model)
 * feature learning instead of feature engineering

* contra RNN
 * BOW is computationally lighter than the RNN approach
 * increasing the size of the hidden state will cube the number of parameters in the model (affects hidden-hidden weights, input-hidden weights and hidden-output weights), the model will therefore have a lot more parameters than a BOW approach
 * more parameters also means that more data will be necessary in order to effectively train the RNN-based classifier

2. x as the final hidden state or the average of all hidden states?
* the final hidden state represents an aggregate of the entire input sequence, depending on the exact RNN implementation this representation might be biased towards the more recent inputs of the sequence
* averaging all hidden states makes sure that the information of all time steps is considered with equal importance, however since every hidden state (h_t), in theory, containts information about all previous hidden states this approach can instead be biased towards earlier inputs in the sequence

3. vanilla RNN vs GRU vs LSTM?
* vanilla RNNs will have issues backpropagating through longer sequences due to exploding/vanishing gradients

4. what happens if you use a bidirectional RNN?
* bidirectional RNNs build their hidden state not only from all past inputs up until t but also from all future inputs from t on, every h_t will therefore be a more balanced representation of the overall input text instead of just at timesteps <=t
* of course this only makes sense when you then create your output layer from a combination of all hidden states and not just the last one

### results

So far I haven't really gotten it to work at all. The accuracy gets stuck at around ~60% which is basically not significantly better than random choice for this dataset. Quite curiously, if you print the histogram of all predictions on the test set, the overall distribution of predictions resembles the ground truth histogram very closely.

![predictions and ground truth histogram on test set][p3_right_histo_wrong_acc.png]

As you can see, the model has certainly learned to predict the different classes roughly in the right distribution, even though it is often wrong on the per-example basis. I am currently still trying to figure out what this might point towards.

## task2: RNN language model
### questions

1. would a different preprocessing make perplexity ratings incomparable?
* the described preprocessing steps (more words as UNK, lowercase everything etc.) would remove variance from the dataset and would therefore also change the evaluated perplexity of the resulting model trained on that data
* if you keep on shrinking your vocabulary size your expected perplexity will also shrink because you can be more certain about the next word just because there are fewer choices
* hypothetically, if you replaced all tokens with UNK the model could not be perplexed at all anymore because it would know with a 100% certainty that the next word would be UNK

2. treat sentences as i.i.d. (inpendent and identically distributed) vs. TBPTT (truncated backprop through time) for training?
* i.i.d. means that each random variable (e.g. a sentence) has the same probability distribution and all of these variables are independent from each other, this is of course a strong simplification because a sentence usually depends on what has been said before and some sentences will certainly appear more often than others
* TBPTT makes similar assumptions by saying that a gradient can not flow more than x steps back, i.e. any hidden state can only depend on the previous x inputs
* in practice TBPTT is very useful for training because inputs will be partitioned into sequences of equal length which makes batching very easy to implement
* the resulting perplexities of both approaches will depend on the average sentence length and the sequence length chosen for TBPTT, if you choose a sequence length that is longer than the average sentence length, TBPTT will likely result in a better language model than the i.i.d. approach

3. character-level RNN language models
* the per-word perplexities of a char-based model and a word-based model should theoretically be comparable, while a char-based model predicts char-by-char (with much higher probabilities because there are fewer possible chars than there are possible words in a word-based model) the resulting probability of predicting a certain word (product of the probabilities of the chars needed to build that word) should be the same as in the word-based model. However, the practical limitations of training on arbitrarily long sequences should result in different per-word perplexities for char-based and word-based models because a char-based model will predict conditioned on the previous x chars and a word-based model on the previous x words. Therefore the word-based model should have more context to base the word prediction on and thus result in a lower perplexity.
* pro char-based RNN
 * much smaller vocabulary size (basically 26 letters + punctuation), therefore predictions will be made over ~50 variables instead of several tens of thousands which will result in a much faster softmax evaluation of the output layer
 * you might potentially be able to learn sub-word level correlations, e.g. dislike/disallow/dismiss
* contra char-based RNN
 * gradient has to flow much further for modeling the same context dependency, therefore it might not learn long-term dependencies as effectively as a word-based model
 * can't use word level correlations as part of your model anymore (i.e. wordvectors)

4. vanilla RNN vs GRU vs LSTM?

5. can you use bidirectional RNNs for language modeling?
* no, this would be a classix example for leakage (using future information to predict the current state)
* a language model is defined as the probability distribution of the current word conditioned on all **previous** words and does not incorporate future information
