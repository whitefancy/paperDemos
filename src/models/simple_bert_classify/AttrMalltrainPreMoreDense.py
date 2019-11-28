from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs

relative_path='/home/zju/buwenfeng/bert/chinese_L-12_H-768_A-12/';
config_path =relative_path+ 'bert_config.json'
checkpoint_path = relative_path+ 'bert_model.ckpt'
dict_path =relative_path+'vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)

import numpy as np
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

maxlen = 180
class data_generator:
    def __init__(self, data, batch_size=16):
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
            np.random.shuffle(self.data)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                attr = d[1]
                point =d[2]
                indices, segments = tokenizer.encode(first=text, second=attr, max_len=maxlen+10)
                X1.append(indices)
                X2.append(segments)
                Y.append([point])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
x = bert_model([x1_in, x2_in])
x=Dense(128,activation='relu')(x)
x=Dropout(0.5)(x) # Dropout正则化
x =Dense(32,kernel_regularizer=regularizers.l2(0.001),activation='relu')(x)
x=Dropout(0.5)(x)
x =Dense(16,activation='relu')(x)
x=Dropout(0.5)(x)
p = Dense(5, activation='softmax')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(3e-5),  # 因为参数少了，增大学习率
    metrics=['accuracy']
)
model_path='/home/zju/buwenfeng/data/sentiment/trainPre增加多层1652/my_model.h5'
model.load_weights(model_path)
import pandas as pd
data_path="/home/zju/buwenfeng/data/mallRatings/"
ratings_len = 10000
import math
from keras import utils
def mergePoint(comment_rating,comments,rating,attr):
    for i in range(len(comments)):
        if((not isinstance(comments[i],float) )and (not math.isnan(rating[i]))):
            one_hot_rating = utils.to_categorical(rating[i]-1,5)
            comment_rating.append((comments[i],attr, one_hot_rating))
#如果没有提到，则评价3分
# ('里面有一些进口食品还是不错的，但个人感觉商品种类比较少，而且管理不是很灵活，退货比较麻烦。价格方面是比较适中的。\n现在没有会员也可以进了，竞争越来越激烈了嘛。\n', '环境', 3.0)

def process_csv_to_data(csv_path,ratings_len):
    pd_ratings = pd.read_csv(csv_path)
    comments = pd_ratings.comment[0:ratings_len]
    rating = pd_ratings.rating[0:ratings_len]
    rating_env = pd_ratings.rating_env[0:ratings_len]
    rating_flavor = pd_ratings.rating_flavor[0:ratings_len]
    rating_service = pd_ratings.rating_service[0:ratings_len]
    comment_rating = []
    mergePoint(comment_rating, comments, rating_env, '环境')
    mergePoint(comment_rating, comments, rating_flavor, '口味')
    mergePoint(comment_rating, comments, rating_service, '服务')
    return comment_rating

comment_rating=process_csv_to_data(data_path+'train40000.csv',ratings_len)
random_order = range(len(comment_rating))
np.random.shuffle(comment_rating)
np.random.shuffle(comment_rating)
np.random.shuffle(comment_rating)
train_data = [comment_rating[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [comment_rating[j] for i, j in enumerate(random_order) if i % 10 == 0]

train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

import keras
import datetime,os
time_id=str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
save_path='/home/zju/buwenfeng/data/sentiment/trainPre增加多层'+time_id+"/";
if not os.path.exists(save_path):
    os.mkdir(save_path)

callbacks_list =[
    keras.callbacks.EarlyStopping(
        monitor='acc',
        patience=1,#大于1轮，也就是两轮没进步就停下来
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=save_path+'my_model.h5',
        monitor='val_loss',#val loss 没有改变就不需要覆盖模型
        save_best_only=True,
    )
]

history = model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=30,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D),
    callbacks=callbacks_list)
