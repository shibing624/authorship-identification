'''1.concat train data and test data 
   2.use lr to fill null label'''

import pandas as pd
import cfg
import json

label_dict = {'人类作者': 0,
              '机器作者': 1,
              '机器翻译': 2,
              '自动摘要': 3}

# ----------------------load data--------------------------------
df_tr = []
count = 0
for i, line in enumerate(open(cfg.data_path + 'training_new.txt', encoding='utf-8')):
    row = {}
    line = line.strip()
    parts = json.loads(line)
    try:
        row['Id'] = parts['id']
        row['label'] = label_dict[parts['标签']]
        row['query'] = parts['内容']
        row['type'] = 'train'
        df_tr.append(row)
        count += 1
    except Exception as e:
        print('err:', e, line)
print("train count:", count)
df_tr = pd.DataFrame(df_tr)

df_te = []
count = 0
for i, line in enumerate(open(cfg.data_path + 'testing.txt', encoding='utf-8')):
    line = line.strip()
    parts = json.loads(line)
    row = {}
    try:
        row['Id'] = parts['id']
        row['label'] = 0
        row['query'] = parts['内容']
        row['type'] = 'test'
        df_te.append(row)
        count += 1
    except Exception as e:
        print('err:', line)
print("test count:", count)
df_te = pd.DataFrame(df_te)

print(df_tr.shape)
print(df_te.shape)

print(df_tr['label'].value_counts())

df_all = pd.concat([df_tr, df_te]).fillna(0)
df_all.to_pickle(cfg.data_path + 'all.pkl')

df_all_1k = pd.concat([df_tr.iloc[:900], df_te.iloc[:100]]).fillna(0)
df_all_1k.to_pickle(cfg.data_path + 'all_1k.pkl')

df_all_test = pd.concat([df_tr.iloc[:10000], df_te.iloc[:100]]).fillna(0)
df_all_test.to_pickle(cfg.data_path + 'all_10k.pkl')

# train count: 146341
# test count:   78041
