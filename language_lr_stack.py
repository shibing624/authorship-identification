'''linguistics_feature_lr stack for label'''
import collections

from scipy import sparse
import pandas as pd
import numpy as np
import jieba
import jieba.posseg
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from datetime import datetime
import cfg


def get_sentence_symbol(sentence_symbol_path='data/sentence_symbol.txt'):
    return [word for word in open(sentence_symbol_path, 'r', encoding='utf-8').read().split()]


def linguistics_feature(data_set, word_sep=' '):
    """
    Get Linguistics feature
    词性表：
    n 名词
    v 动词
    a 形容词
    m 数词
    r 代词
    q 量词
    d 副词
    p 介词
    c 连词
    x 标点
    :param data_set:
    :return:
    """
    features = []
    for line in data_set:
        word_pos_list = line.split(word_sep)
        text_feature = _get_text_feature(word_pos_list)
        feature = text_feature
        for pos in ['n', 'v', 'a', 'm', 'r', 'q', 'd', 'p', 'c', 'x']:
            pos_feature, pos_top = _get_word_feature_by_pos(word_pos_list, pos=pos, most_common_num=10)
            feature.extend(pos_feature)
            # print(pos_top)
        for i in range(len(feature)):
            if feature[i] == 0:
                feature[i] = 1e-5
        feature = [float(i) for i in feature]
        features.append(feature)
        if len(feature) < 35:
            print('error', len(feature), line)
    features_np = np.array(features, dtype=float)
    X = sparse.csr_matrix(features_np)
    return X


def _get_word_feature_by_pos(word_pos_list, pos='n', most_common_num=10):
    n_set = sorted([w for w in word_pos_list if w.endswith(pos)])
    n_len = len(n_set)
    n_ratio = float(len(n_set) / len(word_pos_list))
    n_top = collections.Counter(n_set).most_common(most_common_num)
    return [n_len, n_ratio], n_top


def _get_text_feature(word_pos_list):
    features = []
    num_word = len(word_pos_list)  # 词总数
    assert num_word > 0
    features.append(num_word)
    word_list = [w.split('/')[0] for w in word_pos_list]
    sentence_symbol = get_sentence_symbol()
    sentence_list_long = [w for w in word_list if w in sentence_symbol[:6]]  # 长句
    sentence_list_short = [w for w in word_list if w in sentence_symbol]  # 短句
    num_sentence_long = len(sentence_list_long)
    num_sentence_short = len(sentence_list_short)
    word_no_pos_len_list = [len(w.split('/')[0]) for w in word_pos_list]

    num_char = sum(len(w.split('/')[0]) for w in word_pos_list)  # 字总数
    features.append(num_char)
    average_word_len = float(num_char / num_word)
    features.append(average_word_len)  # 单词平均长度

    # 利用collections库中的Counter模块，可以很轻松地得到一个由单词和词频组成的字典。
    len_counts = collections.Counter(word_no_pos_len_list)

    # 1到4字词个数，1到4字词占比
    for i in range(1, 5):
        features.append(_word_count(len_counts, num=i) if len_counts.get(i) else 0)
        features.append(_word_count_ratio(len_counts, num=i, num_word=num_word) if len_counts.get(i) else 0)
    features.append(num_sentence_long if num_sentence_long > 0 else 0)  # 句子数（长句）
    features.append(float(num_char / num_sentence_long) if num_sentence_long > 0 else 0)  # 句子平均字数

    features.append(num_sentence_short if num_sentence_short > 0 else 0)  # 句子数(短句)
    features.append(float(num_char / num_sentence_short) if num_sentence_short > 0 else 0)  # 句子平均字数（短句）
    return features


def _word_count(counter, num=1):
    """
    1字词个数
    :param counter:
    :param num:
    :return:
    """
    return counter.get(num)


def _word_count_ratio(counter, num=1, num_word=1):
    """
    1字词占比
    :param counter:
    :param num:
    :param num_word:
    :return:
    """
    return float(counter.get(num) / num_word)


# -----------------------myfunc-----------------------
def myAcc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)


# -----------------------load data--------------------

df_all = pd.read_pickle(cfg.data_path + 'all_v2.pkl')
df_query = df_all['query']
query_seg = []
count = 0
for i in df_query:
    count += 1
    words = jieba.posseg.cut(i)
    seg_line = ''
    for word, pos in words:
        seg_line += word + '/' + pos + ' '
    query_seg.append(seg_line)
    if count % 1000 == 0:
        print(count)

with open(cfg.data_path + 'query_seg.txt', 'wb')as f:
    pickle.dump(query_seg, f, protocol=pickle.HIGHEST_PROTOCOL)

# feature
X_sp = linguistics_feature(query_seg)
lb = 'label'
ys = {lb: np.array(df_all[lb])}
df_type = df_all['type']
df_train = [i for i in df_type if i == 'train']
df_train_count = len(df_train)

df_stack = pd.DataFrame(index=range(len(df_all)))

print(lb)
TR = df_train_count
num_class = len(pd.value_counts(ys[lb]))
n = 5

X = X_sp[:TR]
y = ys[lb][:TR]
X_te = X_sp[TR:]
y_te = ys[lb][TR:]

stack = np.zeros((X.shape[0], num_class))
stack_te = np.zeros((X_te.shape[0], num_class))
kf = KFold(n_splits=n)
for i, (tr, va) in enumerate(kf.split(y)):
    print('%s stack:%d/%d' % (str(datetime.now()), i + 1, n))
    clf = LogisticRegression()
    clf.fit(X[tr], y[tr])
    y_pred_va = clf.predict_proba(X[va])
    y_pred_te = clf.predict_proba(X_te)
    print('va acc:', myAcc(y[va], y_pred_va))
    stack[va] += y_pred_va
    stack_te += y_pred_te
stack_te /= n
stack_all = np.vstack([stack, stack_te])
for i in range(stack_all.shape[1]):
    df_stack['language_{}_{}'.format(lb, i)] = stack_all[:, i]

df_stack.to_csv(cfg.data_path + 'language_stack_20W.csv', index=None, encoding='utf8')
print(datetime.now(), 'save language stack done!')

# 1k数据结果：
# va acc: 0.775


# all
# va acc: 0.7253314199808665
