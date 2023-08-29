# 导入必要的包
import paddle

import paddle.dataset.imdb as imdb

import paddle.fluid as fluid

import numpy as np

import os

!mkdir -p /home/aistudio/.cache/paddle/dataset/imdb/

!cp /home/aistudio/data/data7990/aclImdb_v1.tar.gz /home/aistudio/.cache/paddle/dataset/imdb/

# 获取数据字典

print("加载数据字典中...")

word_dict = imdb.word_dict()

# 获取数据字典长度

dict_dim = len(word_dict)

print('完成')

数据是以数据标签的方式表示一个句子。

所以每个句子都是以一串整数来表示的，每个数字都是对应一个单词。

数据集就会有一个数据集字典，这个字典是训练数据中出现单词对应的数字标签。


# 获取训练和预测数据

print("加载训练数据中...")

train_reader = paddle.batch(paddle.reader.shuffle(imdb.train(word_dict),

                                                  512),

                            batch_size=128)

print("加载测试数据中...")

test_reader = paddle.batch(imdb.test(word_dict), 

                           batch_size=128)

print('完成')



