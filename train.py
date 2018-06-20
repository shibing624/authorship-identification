# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import os
from sklearn.model_selection import train_test_split

import config
from models.classic_model import get_model
from models.cnn_model import Model
from models.evaluate import eval
from models.feature import label_encoder
from models.feature import tfidf
from models.reader import build_pos_embedding
from models.reader import build_vocab
from models.reader import build_word_embedding
from models.reader import data_reader
from models.reader import load_emb
from models.reader import load_vocab
from models.reader import test_reader
from models.reader import train_reader
from utils.io_utils import dump_pkl
from utils.io_utils import clear_directory


def train_classic(model_type, data_path=None, pr_figure_path=None,
                  model_save_path=None, vectorizer_path=None, col_sep=',',
                  thresholds=0.5, num_classes=2):
    data_content, data_lbl = data_reader(data_path, col_sep)
    # data feature
    data_tfidf, vocab = tfidf(data_content)
    # save data feature
    dump_pkl(vocab, vectorizer_path, overwrite=True)
    # label
    data_label = label_encoder(data_lbl)
    X_train, X_val, y_train, y_val = train_test_split(
        data_tfidf, data_label, test_size=0.2)
    model = get_model(model_type)
    # fit
    model.fit(X_train, y_train)
    # save model
    dump_pkl(model, model_save_path, overwrite=True)
    # evaluate
    eval(model, X_val, y_val, thresholds=thresholds, num_classes=num_classes,
         model_type=model_type, pr_figure_path=pr_figure_path)


def train_cnn():
    # 1.build vocab for train data
    build_vocab(config.train_seg_path, config.word_vocab_path,
                config.pos_vocab_path, config.label_vocab_path)
    word_vocab, pos_vocab, label_vocab = load_vocab(config.word_vocab_path,
                                                    config.pos_vocab_path,
                                                    config.label_vocab_path)
    # 2.build embedding
    build_word_embedding(config.w2v_path, overwrite=True, sentence_w2v_path=config.sentence_w2v_path,
                         word_vocab_path=config.word_vocab_path, word_vocab_start=config.word_vocab_start,
                         w2v_dim=config.w2v_dim)
    build_pos_embedding(config.p2v_path, overwrite=True, pos_vocab_path=config.pos_vocab_path,
                        pos_vocab_start=config.pos_vocab_start, pos_dim=config.pos_dim)
    word_emb, pos_emb = load_emb(config.w2v_path, config.p2v_path)

    # 3.data reader
    words, pos, labels = train_reader(config.train_seg_path, word_vocab, pos_vocab, label_vocab)
    word_test, pos_test = test_reader(config.test_seg_path, word_vocab, pos_vocab, label_vocab)
    labels_test = None

    # clear
    clear_directory(config.model_save_temp_dir)

    # Division of training, development, and test set
    word_train, word_dev, pos_train, pos_dev, label_train, label_dev = train_test_split(
        words, pos, labels, test_size=0.2, random_state=42)

    # init model
    model = Model(config.max_len, word_emb, pos_emb, label_vocab=label_vocab)
    # fit model
    model.fit(word_train, pos_train, label_train,
              word_dev, pos_dev, label_dev,
              word_test, pos_test, labels_test,
              config.batch_size, config.nb_epoch, config.keep_prob,
              config.word_keep_prob, config.pos_keep_prob)
    [p_test, r_test, f_test], nb_epoch = model.get_best_score()
    print('P@test:%f, R@test:%f, F@test:%f, num_best_epoch:%d' % (p_test, r_test, f_test, nb_epoch + 1))
    # save best pred label
    cmd = 'cp %s/epoch_%d.csv %s/best.csv' % (config.model_save_temp_dir, nb_epoch + 1, config.model_save_dir)
    print(cmd)
    os.popen(cmd)
    # save best model
    cmd = 'cp %s/model_%d.* %s/' % (config.model_save_temp_dir, nb_epoch + 1, config.model_save_dir)
    print(cmd)
    os.popen(cmd)
    # clear model
    model.clear_model()


if __name__ == '__main__':
    if config.model_type != 'cnn':
        train_classic(config.model_type,
                      config.train_seg_path,
                      config.pr_figure_path,
                      config.model_save_path,
                      config.vectorizer_path,
                      config.col_sep,
                      config.pred_thresholds,
                      config.num_classes)
    else:
        train_cnn()
