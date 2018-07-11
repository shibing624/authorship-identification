'''dbow-nn stack for education/age/gender'''

from datetime import datetime

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.cross_validation import KFold

import cfg

# -----------------------load dataset----------------------
df_all = pd.read_pickle(cfg.data_path + 'all_v2.pkl')

lb = 'label'
ys = {lb: np.array(df_all[lb])}
feat = 'dbowd2v'

model = Doc2Vec.load(cfg.data_path + 'dbow_d2v.model')
X_sp = np.array([model.docvecs[i] for i in range(len(df_all))])

# ----------------------dbowd2v stack for label---------------------------
df_stack = pd.DataFrame(index=range(len(df_all)))
df_type = df_all['type']
df_train = [i for i in df_type if i == 'train']
df_train_count = len(df_train)
TR = df_train_count
print('df_train_count:', df_train_count)
n = 5

X = X_sp[:TR]
X_te = X_sp[TR:]

num_class = len(pd.value_counts(ys[lb]))
print("num_class:", num_class)
y = ys[lb][:TR]

stack = np.zeros((X.shape[0], num_class))
stack_te = np.zeros((X_te.shape[0], num_class))
for k, (tr, va) in enumerate(KFold(len(y), n_folds=n)):
    print('{} stack:{}/{}'.format(datetime.now(), k + 1, n))
    nb_classes = num_class
    X_train = X[tr]
    y_train = y[tr]
    X_val = X[va]
    y_val = y[va]

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)

    model = Sequential()
    model.add(Dense(300, input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.25))
    model.add(Activation('tanh'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, shuffle=True,
              batch_size=128, nb_epoch=15,
              verbose=2, validation_data=(X_val, Y_val))
    y_pred_va = model.predict_proba(X_val)
    y_pred_te = model.predict_proba(X_te)
    stack[va] += y_pred_va
    stack_te += y_pred_te
stack_te /= n
stack_all = np.vstack([stack, stack_te])
for l in range(stack_all.shape[1]):
    df_stack['{}_{}_{}'.format(feat, lb, l)] = stack_all[:, l]
df_stack.to_csv(cfg.data_path + 'dbowd2v_stack_20W.csv', encoding='utf8', index=None)
print(datetime.now(), 'save dbowd2v stack done!')


# 1k数据结果：
# va acc: 0.7125
# te acc: 0.78


# 10k:
# val_acc: 0.8394

# all:
# va acc: 0.8459751264179308

# last:
# val_acc: 0.8373
