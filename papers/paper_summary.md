# Intent Clustering Paper Summary

## 1. [SBERT-WK: A Sentence Embedding Method By Dissecting BERT-based Word Models](https://github.com/waynewu6250/Multi-intent-dialoguer/blob/master/papers/BERT-Kuo.pdf)

**Unified word representation:**

1. Weighted averaging vectors across different layers in BERT

2. Inverse alignment weight: <br>
Weight is calculated by defining context window; and summing up all similarity score between current vector and context vector. If the score is high, the weight for it is low, since it contains similar information compared to others.

3. Novelty weight: <br>
Take out the orthogonal vector from the current vector that are not associated with context vector. Decompose Context matrix by SVD and take its column matrix U as basis.
>
    x_orthogonal = x - U * U^T * x

4. Using weight average to combine inverse alignment weight and novelty weight to produce final weight

**Ｗord Importance**

1. Compute cosine similarity matrix between word vectors

2. Extract the offset-1 diagonal of the cosine similarity matrix of that word and compute its variance. The weight for that word is the normalized variance.

3. Final sentence embedding vectors will be the weighted averaging of these weighted word vectors.






