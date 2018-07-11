'''tfidf-lr stack for education/age/gender'''

import pickle
from datetime import datetime

import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

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


class Char_Ngram_Tokenizer():
    def __init__(self):
        self.n = 0

    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = list(query)
            for gram in [1, 2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i + gram])]
        if np.random.rand() < 0.00001:
            print(line)
            print('=' * 20)
            print(tokens)
        self.n += 1
        if self.n % 1000 == 0:
            print(self.n)
        return tokens


class Char_Tokenizer():
    def __init__(self):
        self.n = 0

    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = list(query)
            for gram in [1]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i + gram])]
        if np.random.rand() < 0.00001:
            print(line)
            print('=' * 20)
            print(tokens)
        self.n += 1
        if self.n % 1000 == 0:
            print(self.n)
        return tokens


class Tokenizer():
    def __init__(self):
        self.n = 0

    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = [word for word in jieba.cut(query)]
            for gram in [1, 2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i + gram])]
        if np.random.rand() < 0.00001:
            print(line)
            print('=' * 20)
            print(tokens)
        self.n += 1
        if self.n % 1000 == 0:
            print(self.n)
        return tokens


vectorizer = TfidfVectorizer(tokenizer=Char_Ngram_Tokenizer(), sublinear_tf=True)
X_sp = vectorizer.fit_transform(df_all['query'])
print("feature set nums: ", len(vectorizer.vocabulary_))
feature_names = vectorizer.get_feature_names()

ch2_precent = SelectPercentile(chi2, percentile=80)
ch2 = ch2_precent.fit(X_sp[:df_train_count], df_all.iloc[:df_train_count]['label'])
X_sp = ch2_precent.transform(X_sp)

features = [feature_names[i] for i in ch2.get_support(indices=True)]
feature_scores = [ch2.scores_[i] for i in ch2.get_support(indices=True)]
sorted_feature = sorted(zip(features, feature_scores), key=lambda x: x[1], reverse=True)
feature_output_file = cfg.data_path + 'feature.txt'
with open(feature_output_file, "w", encoding="utf-8") as f_out:
    for id, item in enumerate(sorted_feature):
        f_out.write("\t".join([str(id + 1), item[0], str(item[1])]) + "\n")
print("feature select done,new feature set num: ", len(feature_scores))

pickle.dump(X_sp, open(cfg.data_path + 'tfidf_10W_char.feat', 'wb'))

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
    clf = LogisticRegression()
    # clf = XGBClassifier()
    clf.fit(X[tr], y[tr])
    y_pred_va = clf.predict_proba(X[va])
    y_pred_te = clf.predict_proba(X_te)
    print('va acc:', myAcc(y[va], y_pred_va))
    stack[va] += y_pred_va
    stack_te += y_pred_te
stack_te /= n
stack_all = np.vstack([stack, stack_te])
for i in range(stack_all.shape[1]):
    df_stack['tfidf_lr_{}_{}'.format('label', i)] = stack_all[:, i]

df_stack.to_csv(cfg.data_path + 'tfidf_lr_stack_20W.csv', index=None, encoding='utf8')
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


# last:
#    precision    recall  f1-score   support
#
#           0       0.97      0.96      0.96      9654
#           1       1.00      0.99      0.99      6151
#           2       0.99      1.00      0.99      7261
#           3       0.93      0.95      0.94      6202
#
# avg / total       0.97      0.97      0.97     29268
