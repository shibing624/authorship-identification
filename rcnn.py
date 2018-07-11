# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from datetime import datetime

import keras
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Input, Dense, Conv1D, Embedding, SpatialDropout1D, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import cfg

EMBEDDING_FILE = cfg.data_path + 'wiki_zh.vec'

df_all = pd.read_pickle(cfg.data_path + 'all.pkl')
print('df_all.shape: ', df_all.shape)
df_stack = pd.DataFrame(index=range(len(df_all)))

train = df_all.loc[df_all['type'] == 'train']
test = df_all.loc[df_all['type'] == 'test']
print('df_tr.shape: ', train.shape)

X_train = train["query"]
y_train = train["label"]

X_test = test["query"]
print(X_train.head())
print(y_train[:10])
print(y_train.shape)
print(X_test.head())

max_features = 20000
maxlen = 800
embed_size = 300
batch_size = 512
# epochs = 30
epochs = 10

tok = text.Tokenizer(num_words=max_features)
tok.fit_on_texts(list(X_train) + list(X_test))
X_train = tok.texts_to_sequences(X_train)
X_test = tok.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tok.word_index
# prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print(embedding_matrix)

sequence_input = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
preds = Dense(4, activation="softmax")(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

y_train = keras.utils.to_categorical(y_train, num_classes=4)

print(y_train.shape)
print(y_train[:5])

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train,
                                              train_size=0.9, random_state=0)
print(X_tra.shape)
print(y_tra.shape)
print(X_val.shape)


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))


filepath = cfg.data_path + "rcnn.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
f1_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
callbacks_list = [f1_val, checkpoint, early]

model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs,
          validation_data=(X_val, y_val), callbacks=callbacks_list,
          verbose=1)

model.load_weights(filepath)
print('Predicting....')
y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
y_p = np.argmax(y_pred, 1)

label_revserv_dict = {0: '人类作者',
                      1: '机器作者',
                      2: '机器翻译',
                      3: '自动摘要'}

test['label'] = np.vectorize(label_revserv_dict.get)(y_p)
print(test.head())
test.to_csv(cfg.data_path + 'rcnn_single_result.csv', columns=['Id', 'label'],
            header=False, index=False)

stack_tr = model.predict(X_tra, batch_size=batch_size)
stack_te = model.predict(X_val, batch_size=batch_size)
stack_in = model.predict(x_test, batch_size=batch_size)
stack_all = np.vstack([stack_tr, stack_te, stack_in])
feat = 'rcnn'
lb = 'label'
for l in range(stack_all.shape[1]):
    df_stack['{}_{}_{}'.format(feat, lb, l)] = stack_all[:, l]
df_stack.to_csv(cfg.data_path + 'rcnn_stack_20W.csv', encoding='utf8', index=None)
print(datetime.now(), 'save rcnn_stack_20W.csv done!')
