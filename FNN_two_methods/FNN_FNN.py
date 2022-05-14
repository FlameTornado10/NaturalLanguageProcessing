import json_lines as jls
import re

import matplotlib
import numpy as np
from jedi.api.refactoring import inline
from tqdm import tqdm
import matplotlib.pyplot as plt
np.random.seed(42)

TRAIN = " "
if TRAIN == "test":
    train_file = 'contents_test.jl'
    weight_dir = 'Weights_fnn_small'
else:
    train_file = 'contents_small.jl'
    weight_dir = 'Weights_fnn'
print("train_file: ", train_file)
def tokenize(t):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(t.lower())
def cut_words(file):
    words = []
    with open(file, 'rb') as f:
        for i, item in enumerate(jls.reader(f)):
            if i > 3000:
                break
            words.append(item['p'])
    return words
WORDS_LIMIT = 1000
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
np.save(f'{weight_dir}/dict_fnn.npy',word_to_id)
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
    # print("n_tokens: ", n_tokens)
    for i in range(0, n_tokens - 4):
        target_token = tokens[i + 4]
        for j in range(i, i + 4):
            if tokens[j] in word_to_id:
                temp_token = tokens[j]
            else:
                temp_token = "<unk>"
            X.append(one_hot_encode(word_to_id[temp_token.lower()],
                                len(word_to_id)))
        if target_token in word_to_id:
            y.append(one_hot_encode(word_to_id[target_token.lower()],
                                len(word_to_id)))
        else:
            y.append(one_hot_encode(word_to_id["<unk>".lower()],
                                    len(word_to_id)))
    return np.asarray(X), np.asarray(y)
def generate_test_data(tokens, word_to_id):
    X = []
    for i in range(0, 4):
        X.append(one_hot_encode(word_to_id[tokens[i].lower()],
                                len(word_to_id)))
    return np.asarray(X)
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
    for i, array in enumerate(z):
        z[i] = array + len(array)*[0.0001]
    return - np.sum(np.log(z) * y)
def get_aver_matrix(k1):
    AVER = np.array([[0]*k1*4]*k1)
    TEMP = []
    for i, row in enumerate(AVER):
        child_TEMP = list(row)
        child_TEMP[4*i] = 0.25
        child_TEMP[4*i+1] = 0.25
        child_TEMP[4*i+2] = 0.25
        child_TEMP[4*i+3] = 0.25
        TEMP.append(np.array(child_TEMP))
    AVER = np.array(TEMP)
    return AVER

def forward(model, X, return_cache=True):
    cache = {}
    cache["a1"] = X @ model["w1"]
    k1 = int(X.shape[0]/4)
    AVER = get_aver_matrix(k1)
    cache["b"] = AVER @ cache["a1"]
    cache["a2"] = cache["b"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    if not return_cache:
        print("W1.shape: ",model["w1"].shape)
        print("A1.shape: ",cache["a1"].shape)
        print("B.shape: ",cache["b"].shape)
        print("A2.shape: ",cache["a2"].shape)
        print(type(cache["z"]))
        return cache["z"]
    return cache

def backward(model, X, y, alpha):
    cache  = forward(model, X)
    da2 = cache["z"] - y
    dw2 = cache["b"].T @ da2
    db = da2 @ model["w2"].T
    k1 = int(X.shape[0]/4)
    AVER = get_aver_matrix(k1)
    da1 = AVER.T @ db
    dw1 = X.T @ da1
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

epoch = 20
window = 2
lr = 0.001
history = []
for e in tqdm(range(epoch)):
    for sentence in tqdm(sentences):
        tokens = tokenize(sentence)
        if len(tokens) < 5:
            continue
        X, y = generate_training_data(tokens, word_to_id, window)
        if X.shape[0] == 0:
            continue
        history.append(backward(model, X, y, lr))
    if e % 10 == 0:
        np.save(f'{weight_dir}/weight_1_{e}.npy', model["w1"])
        np.save(f'{weight_dir}/weight_2_{e}.npy', model["w2"])
plt.style.use("seaborn")
plt.plot(range(len(history)), history, color="skyblue")
plt.savefig(f'{weight_dir}/History.png')
plt.show()

learning = generate_test_data(["rich","cultural","and","natural"], word_to_id)
learn = np.array(learning)
print("learn shape", learn.shape)
result = forward(model, learn, return_cache=False)[0]
print("result shape", result.__len__())
id = 0
temp = 0
print(result)
print(np.sum(result))
print(word_to_id)
print(id_to_word)
for i, prob in enumerate(result):
    if temp == 0:
        temp = prob
        id = i
    elif prob > temp:
        temp = prob
        id = i
print("most possible: ",id,result[id])
print(id_to_word[id])