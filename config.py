# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
# data
import os

train_seg_path = "data/training_seg_sample.txt"  # segment of train file
test_seg_path = "data/training_seg_sample.txt"  # segment of test file

pr_figure_path = "output/R_P.png"  # precision recall figure
model_save_path = "output/model.pkl"  # save model path
vectorizer_path = "output/tfidf_vectorizer.pkl"
col_sep = '\t'  # separate label and content of train data

pred_save_path = "output/validation_seg_result.txt"  # infer data result
pred_thresholds = 0.5
num_classes = 4  # num of data label classes

# one of "logistic_regression or random_forest or gbdt or bayes or decision_tree or svm or knn or cnn"
model_type = "cnn"
model_save_dir = "output"

# --- train_w2v_model ---
# path of train sentence, if this file does not exist,
# it will be built from train_seg_path data by train_w2v_model.py train
# word2vec bin path
sentence_w2v_bin_path = model_save_dir + "/sentence_w2v.bin"
# word_dict saved path
sentence_w2v_path = model_save_dir + "/sentence_w2v.pkl"

# --- train ---
word_vocab_path = model_save_dir + "/word_vocab.pkl"
pos_vocab_path = model_save_dir + "/pos_vocab.pkl"
label_vocab_path = model_save_dir + "/label_vocab.pkl"
word_vocab_start = 2
pos_vocab_start = 1

# embedding
w2v_path = model_save_dir + "/w2v.pkl"
p2v_path = model_save_dir + "/p2v.pkl"  # pos vector path
w2v_dim = 256
pos_dim = 64

# param
max_len = 300  # max len words of sentence
min_count = 3  # word will not be added to dictionary if it's frequency is less than min_count
batch_size = 128
nb_epoch = 5
keep_prob = 0.5
word_keep_prob = 0.9
pos_keep_prob = 0.9

# directory to save the trained model
# create a new directory if the dir does not exist
model_save_temp_dir = "output/temp_output_model"
best_result_path = model_save_dir + "/best_result.csv"

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
if not os.path.exists(model_save_temp_dir):
    os.mkdir(model_save_temp_dir)