import json_lines as jls
import re

import matplotlib
import numpy as np
from jedi.api.refactoring import inline
from tqdm import tqdm
np.random.seed(42)
WORDS_LIMIT = 1000
TRAIN = " "
if TRAIN == "test":
    train_file = 'contents_test.jl'
    weight_dir = 'Weights_w2v_small'
else:
    train_file = 'contents_small.jl'
    weight_dir = 'Weights_w2v'
print("train_file: ", train_file)
def tokenize(t):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(t.lower())
def cut_words(file):
    words = []
    with open(file, 'rb') as f:
        for i, item in enumerate(jls.reader(f)):
            # print(item['p'])
            if i > 3000:
                break
            words.append(item['p'])
    return words
def mapping(sentences):
    word_to_id = {}
    id_to_word = {}
    count = 0
    for sentence in sentences:
        tokens = tokenize(sentence)
        for token in set(tokens):
            if token not in word_to_id.keys():
                word_to_id[token.lower()] = count
                id_to_word[count] = token.lower()
                count = count + 1
                if count == WORDS_LIMIT:
                    return word_to_id, id_to_word
    return word_to_id, id_to_word

sentences = cut_words(train_file)
word_to_id, id_to_word = mapping(sentences)
print("words num: ",word_to_id.__len__() + 1)   # <unk> takes 1
word_to_id["<unk>"]=1000
id_to_word[1000]="<unk>"
np.save(f'{weight_dir}/dict_w2v.npy',word_to_id)

def get_range(*iterables):
    for iterable in iterables:
        yield from iterable
def one_hot_encode(id, vocab_size, s=False):
    res = [0] * vocab_size
    res[id] = 1
    if s==True:
        print(len(res))
    return res
def generate_training_data(tokens, word_to_id, window):
    X = []
    y = []
    n_tokens = len(tokens)
    count = 0
    for i in range(n_tokens):
        idx = get_range(
            range(max(0, i - window), i),
            range(i, min(n_tokens, i + window + 1))
        )
        for j in idx:
            if i == j:
                continue
            count = count + 1
            if not tokens[i] in word_to_id:
                tokens[i] = "<UNK>"
            if not tokens[j] in word_to_id:
                tokens[j] = "<UNK>"
            X.append(one_hot_encode(word_to_id[tokens[i].lower()],
                                    len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j].lower()],
                                    len(word_to_id)))
    return np.asarray(X), np.asarray(y)

embedding_num = 10
model = {
    "w1": np.random.randn(len(word_to_id), embedding_num),
    "w2": np.random.randn(embedding_num, len(word_to_id))
}
def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x - np.max(x))
        res.append(exp / exp.sum())
    return res
def cross_entropy(z, y):
    # print(z[0])
    temp = z
    for i, array in enumerate(z):
        z[i] = array + len(array)*[0.0001]
    return - np.sum(np.log(z) * y)
def forward(model, X, return_cache=True):
    cache = {}
    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    if not return_cache:
        return cache["z"]
    return cache
def backward(model, X, y, alpha):
    cache  = forward(model, X)
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

epoch = 20
window = 2
lr = 0.001
history = []
print("weight_dir: ", weight_dir)
for e in tqdm(range(epoch)):
    for sentence in (sentences):
        tokens = tokenize(sentence)
        X, y = generate_training_data(tokens, word_to_id, window)
        if X.shape[0] == 0:
            continue
        history.append(backward(model, X, y, lr))
    if e % 10 == 0:
        np.save(f'{weight_dir}/weight_1_{e}.npy', model["w1"])
        np.save(f'{weight_dir}/weight_2_{e}.npy', model["w2"])
import matplotlib.pyplot as plt
plt.style.use("seaborn")
plt.plot(range(len(history)), history, color="skyblue")
plt.savefig(f'{weight_dir}/History.png')
plt.show()
print(type(word_to_id))
print("len: ",len(word_to_id))
learning = one_hot_encode(word_to_id["learning"], len(word_to_id),s=True)
result = forward(model, np.asarray([learning]), return_cache=False)[0]
for word in (id_to_word[id] for i, id in enumerate(np.argsort(result)[::-1]) if i < 10):
    print(word)