# Word2Vec

## Skip-grams Algorithm

**Skip-grams (SG)** algorithm to predict context words given a target word. This algorithm is independent of the relative positions (offset) of context words and the target word. For example, given a sentence "I like natural language processing." and a window size of 2 words, the algorithm does not discriminate p(natural|processing) with P(language|processing) itself.

Things we need:
* V (1-by-1): number of words in the vocabulary (eg, 10000)
* d (1-by-1): number of features to describe words (eg, 300)
* W_center (V-by-d): word weight matrix (eg, 10000-by-300) with each row as the weight vector for each word when the word is used as a center word
* W_context (V-by-d): word weight matrix (eg, 10000-by-300) with each row as the weight vector for each word when the word is used as a context word, this matrix is different from the center word weight matrix
* input vector w_c (V-by-1): a **one-hot vector** to represent each word in the vocabulary. For the V elements in the vector, only one element is 1, and all other elements are 0. The input matrix for each word is different.
* output vector (V-by-1): a vector with V float elements to represent p(w|w_c) (the probability that a randomly selected nearby word around the word w_c is that word) for all V words. 

Known: V, d, input vector

Unknown: W_center, W_context, output vector

What we need to do is to train the two weight matrices according to certain rules, and once we get the weight matrices, we are able to calculate the output vector for each center word according to the **softmax form**.

### From weight matrices to output vectors

Firstly, assume we have already obtained the weight matrices W_center and W_context, the softmax form tells us about how to compute the output vector.

Softmax:

![softmax](images/lecture2/softmax.png)

In this algorithm, there are 2 word vectors for each word, namely, **center vector** and **context vector**.
* v_c: word vector associated with the center word
* u_o: word vector associated with word at index o

*Note: all indices here are for the vocabulary, i.e, some number out of V, not necessarily correlated with positions of the word in the resource data.*

v_c = (W_center)'*w_c (d-by-1, the transpose of the center weight matrix times the input matrix of the center word). In other words, this is to get the 1-by-d row in the weight matrix for the word and then transpose it to d-by-1. The d elements in the vector represent the weights for the center word on d features.

u_x = (W_context)'*w_x(d-by-1, the transpose of the context weight matrix times the input matrix of the context word). Same as for v_c, the d elements in the vector represent the weights for the context word on d features. For the softmax formula, we need the weight vector for all V context words, while the special one is u_o.

Then we can plug in v_c and u_x's into the softmax formula, and get a V-by-1 vector with each float element as exp(u_o'v_c)/sum(u_x'v_c).

As mentioned at the beginning, the algorithm is indifferent to the offset of the context word relative to the center word. From the output vector, we only know the probability that each word in the vocabulary occurs around a window of size m of the center word. We have no idea about which word will occur on which position (eg, t-1 or t+5).

### Optimize for the weight matrices

The next question for us is, how to get the weight matrices W_center and W_context. One naive and basic training method is to use the softmax form to minimize the loss function.

![optimization1](images/lecture2/optimization1.png)

![optimization2](images/lecture2/optimization2.png)

![optimization3](images/lecture2/optimization3.png)

![optimization4](images/lecture2/optimization4.png)

After we plug the conditional probability function p(o|c) into the **objective/loss function**, we are able to derive it on both v_c and u_o for all V words (i.e., all parameters in the model) and get the V-by-1 gradient of the loss function on the v_c vector, and the V-by-1 gradient of the loss function on the u_o vector. With 2 gradients, we are able to do **gradient descent** and get the optimized values of v_c and u_o for each word. The V 1-by-d vectors for v_c make up the rows of W_center, and the V 1-by-d vectors for u_o make up the rows of W_context.

### Gradient descent

### Hyper-parameters

In this algorithm, all paramters we focus on optimization are the center weight matrix and the context weight matrix, in other words, the V 1-by-d center word vectors and V 1-by-d context word vectors. But there are other hyperparameters we can adjust to get different model performances.

* m: window size
* d: number of features
