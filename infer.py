# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import time

from sklearn.feature_extraction.text import TfidfVectorizer

import config
from models.cnn_model import Model
from models.reader import data_reader
from models.reader import test_reader
from utils.io_utils import load_pkl
from utils.tensor_utils import get_ckpt_path

label_revserv_dict = {0: '人类作者',
                      1: '机器作者',
                      2: '机器翻译',
                      3: '自动摘要'}


def infer_classic(model_save_path, test_data_path, thresholds=0.5,
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
        if test_ids and len(test_ids) > 0:
            assert len(test_ids) == len(label_pred)
            with open(pred_save_path, 'w', encoding='utf-8') as f:
                for i in range(len(label_pred)):
                    f.write(str(test_ids[i]) + ',' + label_revserv_dict[label_pred[i]] + '\n')
        else:
            with open(pred_save_path, 'w', encoding='utf-8') as f:
                for i in range(len(label_pred)):
                    f.write(str(label_pred[i]) + ',' + label_revserv_dict[label_pred[i]] + '\n')
        print("pred_save_path:", pred_save_path)


def infer_cnn(data_path, model_path,
              word_vocab_path, pos_vocab_path, label_vocab_path,
              word_emb_path, pos_emb_path, batch_size, pred_save_path=None):
    # init dict
    word_vocab, pos_vocab, label_vocab = load_pkl(word_vocab_path), load_pkl(pos_vocab_path), load_pkl(label_vocab_path)
    word_emb, pos_emb = load_pkl(word_emb_path), load_pkl(pos_emb_path)
    word_test, pos_test = test_reader(data_path, word_vocab, pos_vocab, label_vocab)
    # init model
    model = Model(config.max_len, word_emb, pos_emb, label_vocab=label_vocab)
    ckpt_path = get_ckpt_path(model_path)
    if ckpt_path:
        print("Read model parameters from %s" % ckpt_path)
        model.saver.restore(model.sess, ckpt_path)
    else:
        print("Can't find the checkpoint.going to stop")
        return
    label_pred = model.predict(word_test, pos_test, batch_size)
    save(label_pred, pred_save_path=pred_save_path)
    print("finish prediction.")


if __name__ == "__main__":
    start_time = time.time()
    if config.model_type != 'cnn':
        infer_classic(config.model_save_path,
                      config.test_seg_path,
                      config.pred_thresholds,
                      config.pred_save_path,
                      config.vectorizer_path,
                      config.col_sep,
                      config.num_classes,
                      config.model_type)
    else:
        infer_cnn(config.test_seg_path,
                  config.model_save_temp_dir,
                  config.word_vocab_path,
                  config.pos_vocab_path,
                  config.label_vocab_path,
                  config.w2v_path,
                  config.p2v_path,
                  config.batch_size,
                  config.pred_save_path)
    print("spend time %ds." % (time.time() - start_time))
