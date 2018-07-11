# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: This example demonstrates the use of fasttext for text classification
# Bi-gram : 0.9056 test accuracy after 5 epochs.
from datetime import datetime

import keras
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

import cfg

def vectorize_words(words, word_idx):
    inputs = []
    for word in words:
        inputs.append([word_idx[w] for w in word])
    return inputs


def create_ngram_set(input_list, ngram_value=2):
    """
    Create a set of n-grams
    :param input_list: [1, 2, 3, 4, 9]
    :param ngram_value: 2
    :return: {(1, 2),(2, 3),(3, 4),(4, 9)}
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list by appending n-grams values
    :param sequences:
    :param token_indice:
    :param ngram_range:
    :return:
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    """
    new_seq = []
    for input in sequences:
        new_list = input[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_seq.append(new_list)
    return new_seq


# -----------------------load dataset----------------------
df_all = pd.read_pickle(cfg.data_path + 'all_v2.pkl')
df_stack = pd.DataFrame(index=range(len(df_all)))

df_train = df_all.loc[df_all['type'] == 'train']
df_infer = df_all.loc[df_all['type'] == 'test']
df_train_count = len(df_train)

print('df_train_count:', df_train_count)

X = df_train['query']
y = df_train['label']
x_infer = df_infer['query']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

num_classes = len(pd.value_counts(y))
print("num_class:", num_classes)

ngram_range = 2
max_features = 20000
max_len = 800
batch_size = 256
embedding_dims = 100
epochs = 3
SAVE_MODEL_PATH = cfg.data_path + 'fasttext_multi_classification_model.h5'

print('loading data...')
x_train = [list(i) for i in x_train]
x_test = [list(i) for i in x_test]
x_infer = [list(i) for i in x_infer]
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

sent_maxlen = max(map(len, (x for x in x_train + x_test)))
print('-')
print('Sentence max length:', sent_maxlen, 'words')
print('Number of training data:', len(x_train))
print('Number of test data:', len(x_test))
print('-')
print('Here\'s what a "sentence" tuple looks like (label, sentence):')
print(y_train[0], x_train[0])
print('-')
print('Vectorizing the word sequences...')

print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

vocab = set()
for w in x_train + x_test + x_infer:
    vocab |= set(w)
vocab = sorted(vocab)
vocab_size = len(vocab) + 1
print('Vocab size:', vocab_size, 'unique words')
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
ids_2_word = dict((value, key) for key, value in word_idx.items())

x_train = vectorize_words(x_train, word_idx)
x_test = vectorize_words(x_test, word_idx)
x_infer = vectorize_words(x_infer, word_idx)
if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # n-gram set from train data
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            ng_set = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(ng_set)
    # add to n-gram
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    max_features = np.max(list(indice_token.keys())) + 1
    # augment x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    x_infer = add_ngram(x_infer, token_indice, ngram_range)

    train_mean_len = np.mean(list(map(len, x_train)), dtype=int)
    test_mean_len = np.mean(list(map(len, x_test)), dtype=int)
    infer_mean_len = np.mean(list(map(len, x_infer)), dtype=int)
    print('Average train sequence length: {}'.format(train_mean_len))
    print('Average test sequence length: {}'.format(test_mean_len))
    print('Average infer sequence length: {}'.format(infer_mean_len))
    max_len = infer_mean_len

print('pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
x_infer = sequence.pad_sequences(x_infer, maxlen=max_len)

print(x_train.shape, 'train shape')
print(x_test.shape, 'x test shape')
print(y_test.shape, 'y test shape')
print(x_infer.shape, 'x_infer shape')

print('build model...')
model = Sequential()

# embed layer by maps vocab index into emb dimensions
model.add(Embedding(max_features, embedding_dims, input_length=max_len))
# pooling the embedding
model.add(GlobalAveragePooling1D())
# output multi classification of num_classes
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
model.save(SAVE_MODEL_PATH)
print('save model:', SAVE_MODEL_PATH)
# loss: 0.0032 - acc: 0.9999 - val_loss: 0.1019 - val_acc: 0.9614

stack_tr = model.predict_proba(x_train, batch_size=batch_size)
stack_te = model.predict_proba(x_test, batch_size=batch_size)
stack_in = model.predict_proba(x_infer, batch_size=batch_size)
stack_all = np.vstack([stack_tr, stack_te, stack_in])
feat = 'fasttext'
lb = 'label'
for l in range(stack_all.shape[1]):
    df_stack['{}_{}_{}'.format(feat, lb, l)] = stack_all[:, l]
df_stack.to_csv(cfg.data_path + 'fasttext_stack_20W.csv', encoding='utf8', index=None)
print(datetime.now(), 'save fasttext stack done!')

# 0.97
