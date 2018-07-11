'''xgb-ens for education/age/gender'''

import pandas as pd
import numpy as np
import xgboost as xgb
import cfg
import datetime

label_revserv_dict = {0: '人类作者',
                      1: '机器作者',
                      2: '机器翻译',
                      3: '自动摘要'}


def xgb_acc_score(preds, dtrain):
    y_true = dtrain.get_label()
    y_pred = np.argmax(preds, axis=1)
    return [('acc', np.mean(y_true == y_pred))]


df_lr = pd.read_csv(cfg.data_path + 'tfidf_lr_stack_20W.csv')
df_svm = pd.read_csv(cfg.data_path + 'tfidf_svm_stack_20W.csv')
df_xgblr = pd.read_csv(cfg.data_path + 'tfidf_xgblr_stack_20W.csv')
df_dm = pd.read_csv(cfg.data_path + 'dmd2v_stack_20W.csv')
df_dbow = pd.read_csv(cfg.data_path + 'dbowd2v_stack_20W.csv')
df_fasttext = pd.read_csv(cfg.data_path + 'fasttext_stack_20W.csv')
df_mcnn = pd.read_csv(cfg.data_path + 'mcnn_stack_20W.csv')
df_lstm = pd.read_csv(cfg.data_path + 'lstm_stack_20W.csv')

df_all = pd.read_pickle(cfg.data_path + 'all.pkl')

y = df_all['label']

'''最好的参数组合'''
# -------------------------label----------------------------------
df_type = df_all['type']
df_train = [i for i in df_type if i == 'train']
df_train_count = len(df_train)
TR = df_train_count
df_sub = pd.DataFrame()
df_sub['Id'] = df_all.iloc[TR:]['Id']
lb = 'label'
print(lb)

df = pd.concat([df_lr, df_svm, df_xgblr, df_dbow, df_dm, df_fasttext, df_mcnn, df_lstm], axis=1)
print(df.columns)
num_class = len(pd.value_counts(y))
print('num_class:', num_class)
X = df.iloc[:TR]
y = y.iloc[:TR]
X_te = df.iloc[TR:]
y_te = y.iloc[TR:]

esr = 100
evals = 1
ss = 1
mc = 1
md = 8
gm = 1
n_trees = 30

params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    "num_class": num_class,
    'max_depth': md,
    'min_child_weight': mc,
    'subsample': ss,
    'colsample_bytree': 0.8,
    'gamma': gm,
    "eta": 0.01,
    "lambda": 0,
    'alpha': 0,
    "silent": 1,
}

dtrain = xgb.DMatrix(X, y)
dvalid = xgb.DMatrix(X_te, y_te)
watchlist = [(dtrain, 'train')]
# watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
bst = xgb.train(params, dtrain, n_trees, evals=watchlist, feval=xgb_acc_score, maximize=True,
                early_stopping_rounds=esr, verbose_eval=evals)
pred = np.argmax(bst.predict(dvalid), axis=1)
pred_label = [label_revserv_dict[int(i)] for i in pred]
df_sub['label'] = np.array(pred_label)
df_sub = df_sub[['Id', 'label']]
df_sub.to_csv(cfg.data_path + 'all_result_20W.csv', index=None, header=None, sep=',')


# 1k数据结果：
# train-acc:0.955
# eval-acc:0.86

# all: n_trees:30
# train-acc:0.981188， 线上：0.97548229

# all  tfidf_language_dm_dbow_20W
# train-acc: 0.983545, 线上：0.975357

# all  tfidf_fasttext_dm_dbow_20W
# train-acc: 0.982001, 线上：0.97558
