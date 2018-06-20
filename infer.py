# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from utils.io_utils import load_pkl
from models.reader import data_reader

label_revserv_dict = {0: '人类作者',
                      1: '机器作者',
                      2: '机器翻译',
                      3: '自动摘要'}


def infer(model_save_path, test_data_path, thresholds=0.5,
          pred_save_path=None, vectorizer_path=None, col_sep=',',
          num_classes=2, model_type='svm'):
    # load model
    model = load_pkl(model_save_path)
    # load data content
    data_set, test_ids = data_reader(test_data_path, col_sep)
    # data feature
    vocab = load_pkl(vectorizer_path)
    vectorizer = TfidfVectorizer(analyzer='char', vocabulary=vocab)
    data_feature = vectorizer.fit_transform(data_set)
    if num_classes == 2 and model_type != 'svm':
        # binary classification
        label_pred_probas = model.predict_proba(data_feature)[:, 1]
        label_pred = label_pred_probas > thresholds
    else:
        label_pred = model.predict(data_feature)  # same
    save(label_pred, test_ids, pred_save_path)
    print("finish prediction.")


def save(label_pred, test_ids=[], pred_save_path=None):
    if pred_save_path:
        assert len(test_ids) == len(label_pred)
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            for i in range(len(label_pred)):
                f.write(str(test_ids[i]) + ',' + label_revserv_dict[label_pred[i]] + '\n')
        print("pred_save_path:", pred_save_path)


if __name__ == "__main__":
    infer(config.model_save_path,
          config.test_seg_path,
          config.pred_thresholds,
          config.pred_save_path,
          config.vectorizer_path,
          config.col_sep,
          config.num_classes,
          config.model_type)
