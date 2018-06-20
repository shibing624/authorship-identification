# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2


def tfidf(data_set, is_infer=False):
    """
    Get TFIDF value
    :param data_set:
    :return:
    """
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


def label_encoder(labels):
    encoder = preprocessing.LabelEncoder()
    corpus_encode_label = encoder.fit_transform(labels)
    print('corpus_encode_label shape:', corpus_encode_label.shape)
    return corpus_encode_label


def select_best_feature(data_set, data_lbl):
    ch2 = SelectKBest(chi2, k=10000)
    return ch2.fit_transform(data_set, data_lbl), ch2
