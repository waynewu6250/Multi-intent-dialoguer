<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

# Intent Clustering Paper Summary

## 1. [SBERT-WK: A Sentence Embedding Method By Dissecting BERT-based Word Models](https://github.com/waynewu6250/Multi-intent-dialoguer/blob/master/papers/BERT-Kuo.pdf)

**Unified word representation:**

1. Weighted averaging vectors across different layers in BERT

2. Inverse alignment weight: <br>
Weight is calculated by defining context window; and summing up all similarity score between current vector and context vector. If the score is high, the weight for it is low, since it contains similar information compared to others.

3. Novelty weight: <br>
Take out the orthogonal vector from the current vector that are not associated with context vector. Decompose Context matrix by SVD and take its column matrix U as basis. \\[ x_orthogonal = x - U * U^T * x \\]






