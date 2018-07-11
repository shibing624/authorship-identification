# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: This example demonstrates the use of fasttext for text classification
# Bi-gram : 0.9056 test accuracy after 5 epochs.
from datetime import datetime

import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout, GlobalMaxPooling1D, Activation, MaxPooling1D
from keras.layers import Embedding, Input, concatenate
from keras.layers.convolutional import Convolution1D
from keras.models import Sequential, Model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

import cfg

def vectorize_words(words, word_idx):
    inputs = []
    for word in words:
        inputs.append([word_idx[w] for w in word])
    return inputs


# -----------------------load data set----------------------
df_all = pd.read_pickle(cfg.data_path + 'all.pkl')
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

max_len = 400
batch_size = 256
embedding_dims = 100
epochs = 10
SAVE_MODEL_PATH = cfg.data_path + 'mcnn_multi_classification_model.h5'

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

print('pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
x_infer = sequence.pad_sequences(x_infer, maxlen=max_len)

print(x_train.shape, 'train shape')
print(x_test.shape, 'x test shape')
print(y_test.shape, 'y test shape')
print(x_infer.shape, 'x_infer shape')

print('build model...')
input_x = Input(shape=(max_len,), dtype='int32')
# embed layer by maps vocab index into emb dimensions
emb = Embedding(vocab_size, embedding_dims, input_length=max_len)(input_x)
conv_out = []
for filter_size in [2, 3, 4]:
    x = Convolution1D(256, filter_size, padding='same')(emb)
    x = Activation('relu')(x)
    x = Convolution1D(256, filter_size, padding='same')(x)
    x = Activation('relu')(x)
    x = GlobalMaxPooling1D()(x)
    conv_out.append(x)
x = concatenate(conv_out)
x = Dropout(0.25)(x)
x = Dense(256)(x)
x = Activation('relu')(x)
# output multi classification of num_classes
x = Dense(num_classes)(x)
outputs = Activation('softmax', name='outputs')(x)
model = Model(inputs=[input_x], outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
model.save(SAVE_MODEL_PATH)
print('save model:', SAVE_MODEL_PATH)
# loss: 0.0032 - acc: 0.9999 - val_loss: 0.1019 - val_acc: 0.9614

stack_tr = model.predict(x_train, batch_size=batch_size)
stack_te = model.predict(x_test, batch_size=batch_size)
stack_in = model.predict(x_infer, batch_size=batch_size)
stack_all = np.vstack([stack_tr, stack_te, stack_in])
feat = 'mcnn'
lb = 'label'
for l in range(stack_all.shape[1]):
    df_stack['{}_{}_{}'.format(feat, lb, l)] = stack_all[:, l]
df_stack.to_csv(cfg.data_path + 'mcnn_stack_20W.csv', encoding='utf8', index=None)
print(datetime.now(), 'save mcnn_stack_20W.csv done!')
