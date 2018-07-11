'''tfidf-lr stack for education/age/gender'''

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import pickle
import cfg


# -----------------------myfunc-----------------------
def myAcc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_true, y_pred))
    return np.mean(y_true == y_pred)


# -----------------------load data--------------------
df_all = pd.read_pickle(cfg.data_path + 'all.pkl')
print(df_all.shape)
df_stack = pd.DataFrame(index=range(len(df_all)))
df_type = df_all['type']
df_train = [i for i in df_type if i == 'train']
df_train_count = len(df_train)
TR = df_train_count
print('df_train_count:', df_train_count)

X_sp = pickle.load(open(cfg.data_path + 'tfidf_10W_char.feat', 'rb'))

# -----------------------stack for label------------------
TR = df_train_count
n = 5

X = X_sp[:TR]
y = df_all['label'].iloc[:TR]
X_te = X_sp[TR:]
y_te = df_all['label'].iloc[TR:]
num_class = len(pd.value_counts(y))
stack = np.zeros((X.shape[0], num_class))
stack_te = np.zeros((X_te.shape[0], num_class))
kf = KFold(n_splits=n)
for i, (tr, va) in enumerate(kf.split(y)):
    print('%s stack:%d/%d' % (str(datetime.now()), i + 1, n))
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X[tr], y[tr])
    y_pred_va = clf.predict_proba(X[va])
    y_pred_te = clf.predict_proba(X_te)
    print('va acc:', myAcc(y[va], y_pred_va))
    # print('te acc:', myAcc(y_te, y_pred_te))
    stack[va] += y_pred_va
    stack_te += y_pred_te
stack_te /= n
stack_all = np.vstack([stack, stack_te])
for i in range(stack_all.shape[1]):
    df_stack['tfidf_svm_{}_{}'.format('label', i)] = stack_all[:, i]

df_stack.to_csv(cfg.data_path + 'tfidf_svm_stack_20W.csv', index=None, encoding='utf8')
print(datetime.now(), 'save tfidf stack done!')

# word seg
# 4label
# LR

# 1k数据结果：
# va acc: 0.7625
# te acc: 0.75

# 5w数据结果
# va acc: 0.9622

# all
# va acc: 0.9729748197751887


# 3 label
# 5w数据结果
# va acc: 0.875

# char seg
# 4label
# LR
# all
# va acc: 0.9733848098670949

# char seg
# 4label
# xgb
# all
# va acc: 0.968019680196802

# add feature select
# va acc: 0.9746322263222632
# detail:   0       0.75      0.83      0.79       682
#           1       0.99      0.93      0.96       371
#           2       0.92      0.98      0.95       491
#           3       0.75      0.62      0.68       456
#
# avg / total       0.84      0.84      0.84      2000

# svm add feature
# 2018-07-10 21:04:53.806184 stack:1/5
#              precision    recall  f1-score   support
#
#           0       0.83      0.82      0.83       661
#           1       0.98      0.97      0.97       410
#           2       0.96      0.98      0.97       523
#           3       0.75      0.76      0.75       406
#
# avg / total       0.88      0.88      0.88      2000

# svm no trim feature
# 2018-07-10 21:17:22.300468 stack:1/5
#              precision    recall  f1-score   support
#
#           0       0.82      0.83      0.83       661
#           1       0.98      0.96      0.97       410
#           2       0.96      0.98      0.97       523
#           3       0.75      0.74      0.75       406
#
# avg / total       0.88      0.88      0.88      2000
#
# va acc: 0.877
