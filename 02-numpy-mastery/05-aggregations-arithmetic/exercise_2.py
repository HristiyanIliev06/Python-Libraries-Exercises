import numpy as np

# A:
np.random.seed(42)
# logits = np.random.randn(32, 10)   # 32 samples, 10 classes
# labels = np.random.randint(0, 10, size=32)  # true class indices

def stable_softmax(logits: np.ndarray):

    max_raw_logit = logits.max(axis = 1, keepdims = True)
    logits_shifted = logits - max_raw_logit
    exp_logits = np.exp(logits_shifted)
    softmax = exp_logits / exp_logits.sum(axis = 1, keepdims = True)

    return softmax

def unstable_softmax(logits: np.ndarray):

    exp_logits = np.exp(logits)
    softmax = exp_logits / exp_logits.sum(axis = 1, keepdims = True)

    return softmax

def cross_entropy_loss(probs, labels):
    rows = np.arange(len(labels))
    cols = labels

    log_probs = np.log(probs[rows, cols])
    loss = -np.mean(log_probs)

    return loss

logits = np.array([
    [1000, 1001, 998],
    [5, 2, 1],
    [0.1, 0.2, 0.1]
])
labels = np.array([1, 0, 1])

softmax = stable_softmax(logits)
print(softmax)
print('\n')
print(cross_entropy_loss(softmax, labels))

#print(unstable_softmax(logits))
row_sums = softmax.sum(axis=1)
print(np.allclose(row_sums, np.ones(3)))


