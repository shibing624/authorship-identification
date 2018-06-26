# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import collections

import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.data_utils import load_pkl
from models.reader import get_contents_words, get_sentence_symbol


def tfidf(data_set, is_infer=False):
    """
    Get TFIDF value
    :param data_set:
    :return:
    """
    data_set = get_contents_words(data_set)
    vectorizer = TfidfVectorizer(analyzer='char')
    if not is_infer:
        data_feature = vectorizer.fit_transform(data_set)
    else:
        data_feature = vectorizer.transform(data_set)
    vocab = vectorizer.vocabulary_
    print('Vocab size:', len(vocab))
    print('Vocab list:')
    count = 0
    for k, v in vectorizer.vocabulary_.items():
        if count < 10:
            print(k, v)
            count += 1

    print('\nIFIDF词频矩阵:')
    print('data_feature shape:', data_feature.shape)
    print(data_feature.toarray())
    return data_feature, vocab


def tf(data_set, is_infer=False):
    """
    Get TFIDF value
    :param data_set:
    :return:
    """
    data_set = get_contents_words(data_set)
    vectorizer = CountVectorizer(analyzer='char')
    if not is_infer:
        data_feature = vectorizer.fit_transform(data_set)
    else:
        data_feature = vectorizer.transform(data_set)
    vocab = vectorizer.vocabulary_
    print('Vocab size:', len(vocab))
    print('Vocab list:')
    count = 0
    for k, v in vectorizer.vocabulary_.items():
        if count < 10:
            print(k, v)
            count += 1

    print('\nTF词频矩阵:')
    print('data_feature shape:', data_feature.shape)
    print(data_feature.toarray())
    return data_feature, vocab


def linguistics(data_set, word_sep=' '):
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
                feature[i] = 0.00001
        feature = [float(i) for i in feature]
        features.append(feature)
        if len(feature) < 35:
            print('error', len(feature), line)
    features_np = np.array(features, dtype=float)
    X = sparse.csr_matrix(features_np)
    return X, None


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


def all_human_feature(data_set):
    """
    Get all_human_feature
    :param data_set:
    :return:
    """
    tfidf_feature, vocab = tfidf(data_set)
    tfidf_feature_np = tfidf_feature.toarray()
    linguistics_feature, _ = linguistics(data_set)
    linguistics_feature_np = linguistics_feature.toarray()

    data_feature = np.hstack((tfidf_feature_np, linguistics_feature_np))
    return data_feature, vocab


def label_encoder(labels):
    encoder = preprocessing.LabelEncoder()
    corpus_encode_label = encoder.fit_transform(labels)
    print('corpus_encode_label shape:', corpus_encode_label.shape)
    return corpus_encode_label


def select_best_feature(data_set, data_lbl):
    ch2 = SelectKBest(chi2, k=10000)
    return ch2.fit_transform(data_set, data_lbl), ch2


def get_feature(data_set, feature_type='tf', is_infer=False, infer_vectorizer_path=None):
    if is_infer:
        if feature_type == "tf":
            vocab = load_pkl(infer_vectorizer_path)
            vectorizer = CountVectorizer(analyzer='char', vocabulary=vocab)
            return vectorizer.fit_transform(data_set)
        elif feature_type == "tfidf":
            vocab = load_pkl(infer_vectorizer_path)
            vectorizer = TfidfVectorizer(analyzer='char', vocabulary=vocab)
            return vectorizer.fit_transform(data_set)
        elif feature_type == "linguistics":
            data_feature, _ = linguistics(data_set)
            return data_feature
        elif feature_type == 'all':
            vocab = load_pkl(infer_vectorizer_path)
            vectorizer = TfidfVectorizer(analyzer='char', vocabulary=vocab)
            tfidf_feature_np = vectorizer.fit_transform(data_set).toarray()
            linguistics_feature, _ = linguistics(data_set)
            linguistics_feature_np = linguistics_feature.toarray()

            data_feature = np.hstack((tfidf_feature_np, linguistics_feature_np))
            return data_feature
    else:
        if feature_type == "tf":
            return tf(data_set)
        elif feature_type == "tfidf":
            return tfidf(data_set)
        elif feature_type == "linguistics":
            return linguistics(data_set)
        elif feature_type == 'all':
            return all_human_feature(data_set)
