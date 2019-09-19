#! -*- coding:utf-8 -*-

import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs

# 每个.zip文件包含三个项目：
#
# TensorFlow检查点（bert_model.ckpt）包含预先训练的权重（实际上是3个文件）。
# vocab.txt用于将WordPiece映射到word id的词汇文件（）。
# 配置文件（bert_config.json），指定模型的超参数。
maxlen = 100
relative_path='/home/zju/buwenfeng/bert/chinese_L-12_H-768_A-12/';
config_path =relative_path+ 'bert_config.json'
checkpoint_path = relative_path+ 'bert_model.ckpt'
dict_path =relative_path+  'vocab.txt'


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# 这里简单解释一下Tokenizer的输出结果。首先，默认情况下，分词后句子首位会分别加上[CLS]和[SEP]标记，其中[CLS]位置对应的输出向量是能代表整句的句向量（反正Bert是这样设计的），而[SEP]则是句间的分隔符，其余部分则是单字输出（对于中文来说）。
#
# 本来Tokenizer有自己的_tokenize方法，
# 重写了这个方法，是要保证tokenize之后的结果，
# 跟原来的字符串长度等长（如果算上两个标记，那么就是等长再加2）。
# Tokenizer自带的_tokenize会自动去掉空格，
# 然后有些字符会粘在一块输出，
# 导致tokenize之后的列表不等于原来字符串的长度了，
# 这样如果做序列标注的任务会很麻烦。
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)
#tokenizer.tokenize(u'他是我son')
#Out[15]: ['[CLS]', '他', '是', '我', 's', 'o', 'n', '[SEP]']


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)#token_ids, segment_ids
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

# 注意，尽管可以设置seq_len=None，但是仍要保证序列长度不超过512
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
#model = keras.models.Model(inputs=inputs, outputs=outputs)
x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)# 取出[CLS]对应的向量用来做分类
p = Dense(1, activation='sigmoid')(x)
#“有什么原则来指导Bert后面应该要接哪些层？”。
# 答案是：用尽可能少的层来完成你的任务。
# 比如上述情感分析只是一个二分类任务，
# 你就取出第一个向量然后加个Dense(1)就好了，不
# 要想着多加几层Dense，更加不要想着接个LSTM再接Dense；
# 如果你要做序列标注（比如NER），那你就接个Dense+CRF就好，
# 也不要多加其他东西。总之，额外加的东西尽可能少。
# 一是因为Bert本身就足够复杂，它有足够能力应对你要做的很多任务；
# 二来你自己加的层都是随即初始化的，加太多会对Bert的预训练权重造成剧烈扰动，
# 容易降低效果甚至造成模型不收敛～
model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()

from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 500
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = sequence.pad_sequences(x_test,maxlen=maxlen)

neg = pd.read_excel('neg.xls', header=None)
pos = pd.read_excel('pos.xls', header=None)

data = []

for d in neg[0]:
    data.append((d, 0))

for d in pos[0]:
    data.append((d, 1))


# 按照9:1的比例划分训练集和验证集
random_order = range(len(data))
np.random.shuffle(data)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

#如果你的显存不够大，
# 将句子的maxlen和batch size都调小一点试试。
# 当然，如果你的任务太复杂，
# 再小的maxlen和batch size也可能OOM，那就只有升级显卡了。
history = model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D))


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label = 'Training loss')# bo 表示蓝色圆点
plt.plot(epochs,val_loss,'b',label='Validation loss') # b 表示蓝色实线
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

