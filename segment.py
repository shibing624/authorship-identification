# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import json
from time import time
import jieba
import jieba.posseg

label_dict = {'人类作者': 0,
              '机器作者': 1,
              '机器翻译': 2,
              '自动摘要': 3}


class Tokenizer():
    def __init__(self):
        self.n = 0

    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = [word for word in jieba.cut(query)]
            for gram in [1, 2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i + gram])]
        self.n += 1
        if self.n % 10000 == 0:
            print(self.n, end=' ')
            print(line)
            print('=' * 20)
            print(tokens)
        return tokens


def parse_train_json(in_file, out_file, pos=False):
    with open(in_file, 'r') as f1, open(out_file, 'w', encoding='utf-8') as f2:
        count = 0
        for line in f1:
            line = line.strip()
            parts = json.loads(line)
            label_tag = parts['标签']
            label = label_dict[label_tag]
            data = parts['内容']
            if pos:
                words = jieba.posseg.cut(data)
                seg_line = ''
                for word, pos in words:
                    seg_line += word + '/' + pos + ' '
            else:
                seg_line = ' '.join(jieba.cut(data))
            f2.write(str(label) + '\t' + seg_line + "\n")
            count += 1
        print('%s to %s, size: %d' % (in_file, out_file, count))


def parse_val_json(in_file, out_file, pos=False):
    with open(in_file, 'r') as f1, open(out_file, 'w', encoding='utf-8') as f2:
        count = 0
        for line in f1:
            line = line.strip()
            parts = json.loads(line)
            label = parts['id']
            data = parts['内容']
            if pos:
                words = jieba.posseg.cut(data)
                seg_line = ''
                for word, pos in words:
                    seg_line += word + '/' + pos + ' '
            else:
                seg_line = ' '.join(jieba.cut(data))
            f2.write(str(label) + '\t' + seg_line + "\n")
            count += 1
            if count % 1000 == 0:
                print('count:', count)
        print('%s to %s, size: %d' % (in_file, out_file, count))


if __name__ == '__main__':
    # 训练集，格式：{'标签': '人类作者', '内容': '~===全国性新规===英烈保护法通过！宣扬、美化侵略战争或追刑责山东省枣庄市', 'id': 10595}
    train_file = './data/training.txt'
    # 验证集，格式：{'id': 165484, '内容': '13日夜间到14日白天，全省各地晴或多云。}
    val_file = './data/testing.txt'

    save_train_seg_file = './data/training_seg.txt'
    save_val_seg_file = './data/testing_seg.txt'

    start_time = time()
    parse_train_json(train_file, save_train_seg_file, True)
    parse_val_json(val_file, save_val_seg_file, True)
    print("spend time:", time() - start_time)
