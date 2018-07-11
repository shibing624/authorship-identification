'''train dbow/dm for education/age/gender'''

import codecs
import os
from datetime import datetime

import gensim
import jieba
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import cfg

df_all = pd.read_pickle(cfg.data_path + 'all.pkl')
output_file_path = cfg.data_path + 'all_segmented.txt'
if not os.path.exists(output_file_path):
    with codecs.open(output_file_path, 'w', encoding='utf8') as doc_f:
        for i, query in enumerate(df_all['query']):
            try:
                words = jieba.lcut(query)
                # words = list(query)
            except Exception as e:
                print('seg error: ', i, query)
            if i % 1000 == 0:
                print(datetime.now(), i)
            doc_f.write('{}\n'.format(' '.join(words)))

d2v = Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=3, window=10, sample=1e-5, workers=8, alpha=0.025,
              min_alpha=0.025)
sentences = gensim.models.doc2vec.TaggedLineDocument(output_file_path)
d2v.build_vocab(sentences)

# -------------------train dbow doc2vec---------------------------------------------
df_lb = df_all['label'][:1000]
for i in range(5):
    print(datetime.now(), 'dbow pass:', i + 1)
    d2v.train(sentences, total_examples=d2v.corpus_count, epochs=10)
    X_d2v = np.array([d2v.docvecs[i] for i in range(1000)])
    scores = cross_val_score(LogisticRegression(), X_d2v, df_lb, cv=5)
    print('dbow label', scores, np.mean(scores))
d2v.save(cfg.data_path + 'dbow_d2v.model')
print(datetime.now(), 'save done')

d2v = Doc2Vec(dm=1, size=300, negative=5, hs=0, min_count=3, window=5, sample=1e-5, workers=8, alpha=0.05,
              min_alpha=0.025)
d2v.build_vocab(sentences)

# ---------------train dm doc2vec-----------------------------------------------------
for i in range(5):
    print(datetime.now(), 'dm pass:', i + 1)
    d2v.train(sentences, total_examples=d2v.corpus_count, epochs=10)
    X_d2v = np.array([d2v.docvecs[i] for i in range(1000)])
    scores = cross_val_score(LogisticRegression(), X_d2v, df_lb, cv=5)
    print('dm label', scores, np.mean(scores))
d2v.save(cfg.data_path + 'dm_d2v.model')
print(datetime.now(), 'save done')


# ALL数据作为doc,1k数据结果：
# dbow
# te acc: 0.729
# dm
# te acc: 0.703

# ALL数据作为doc, 1w数据做LR，结果：
# dbow
# te acc:0.809
# dm
# te acc:0.737

# char seg:5轮
# dbow:0.7180
# dm: 0.653


# word seg:5轮
# dbow:0.7631
# dm:0.728


# last:1轮
# all data ,1k test:
# dbow: acc:0.764
# dm: acc:0.648
