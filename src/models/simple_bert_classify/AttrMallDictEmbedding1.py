answer_vacabulary_size=5
save_file_path ='/home/zju/buwenfeng/autosave'
batch_size=64
import keras
import datetime,os
time_id=str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
save_path='/home/zju/buwenfeng/autosave/dict'+time_id+"/";
if not os.path.exists(save_path):
    os.mkdir(save_path)

#DAE 降维
import os
import numpy as np
book_dir = '/home/zju/buwenfeng/TensorFlowDemo/dict_space'
book_name ='chinesedictory5.txt'
embeddings_index = {}
f = open(os.path.join(book_dir,book_name),'r',encoding='utf-16')
words = []
for line in f:
    values = line.split()
    words.append(values)

f.close()
#去掉空行
words = [ x for x in words if len(x)>0]#  len(words) 133060 to 67942
#找到所有的无重复的字符
chars=[] #总字数2658545 去除了空白字符
for x in words:
    chars.extend(list(x[0]))

unichars = list(set(chars))#所有出现的字符数量为12889，也是所有向量的个数
chars=[]
#unichars转成字典
char_ind = np.linspace(1, len(unichars), num=len(unichars))
ciDict = dict((ch,ind)for ch,ind in zip(unichars,char_ind))
#unichars也就是字的索引
#处理成机器学习能计算的浮点数词典
#词典空间的维度等于所有的行数67942
# #所以，词典空间的大小为12889*67942
vec_space= np.zeros([len(unichars),len(words)])
#统计所有token出现的分布矩阵
for ix,tokens in enumerate(words):
    for iy,y in enumerate(list(tokens[0])):
        vec_space[ciDict.get(y)][ix] += 1


v1 =ciDict.get('蜻')
v2 =ciDict.get('得')
v3 =ciDict.get('获')
dist = np.linalg.norm(vec_space[v1] - vec_space[v2])
dist = np.sqrt(np.sum(np.square(vec_space[v1] - vec_space[v2])))
#去掉频率过高的和频率过低的字符
freq_dict =np.array([ ch  for i,ch in enumerate(vec_space) if (np.sum(ch)>3 and np.sum(ch)<10000)])

import numpy as np
from keras.layers import *
from keras.models import Model
import keras
inputDims = vec_space.shape[1]
EncoderDims = 512
x_in =Input(shape=(inputDims,))
x_encode=Dense(EncoderDims,activation='relu',name='encode_layer')(x_in)
x_decode=Dense(inputDims,activation='sigmoid')(x_encode)
AutoEncoder = Model(x_in, x_decode)
AutoEncoder.compile(optimizer='adadelta',loss='binary_crossentropy')
callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=1,  # 大于1轮，也就是两轮没进步就停下来
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=save_path + 'my_model.h5',
                monitor='loss',  # val loss 没有改变就不需要覆盖模型
                save_best_only=True,
            )]
AutoEncoder.fit(freq_dict,freq_dict,batch_size=32,nb_epoch=20,shuffle=True)
encode_layer_weights =AutoEncoder.get_layer('encode_layer').get_weights()
from keras import backend as K
x1_in = Input(shape=(inputDims,))
x2_in = Input(shape=(inputDims,))
x_encode1=Dense(EncoderDims,activation='relu',name='encode1')(x1_in)
x_expand1=Lambda(lambda  x:K.expand_dims(x,axis=-1))(x_encode1)
encoded_text =LSTM(16)(x_expand1)
x_encode2=Dense(EncoderDims,activation='relu',name='encode2')(x2_in)
x_expand2=Lambda(lambda  x:K.expand_dims(x,axis=-1))(x_encode2)
encoded_question = LSTM(8)(x_expand2)
concatenated = concatenate([encoded_text,encoded_question],axis=-1) #将编码后的问题和文本连接起来
answer = Dense(answer_vacabulary_size,activation='softmax')(concatenated)#问题的答案可能只有几个词，所以比起问题，单词数更少
model = Model([x1_in,x2_in],answer) # 模型实例化时，指定两个输入和一个输出
#将预训练的词嵌入加载到Embedding层中
model.get_layer('encode1').set_weights(encode_layer_weights)
model.get_layer('encode2').set_weights(encode_layer_weights)
model.get_layer('encode1').trainable = False #冻结Embedding层 以免丢失保存的信息 否则会被随机初始化
model.get_layer('encode2').trainable = False
model.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta',  # 因为参数少了，增大学习率
    metrics=['accuracy'])

import pandas as pd
data_path="/home/zju/buwenfeng/data/mallratings/"
ratings_len = 20000
import math
from keras.utils import to_categorical
def mergePoint(comment_rating,comments,rating,attr):
    for i in range(len(comments)):
        if((not isinstance(comments[i],float) )and (not math.isnan(rating[i]))):
            one_hot_rating = to_categorical(rating[i]-1,5)
            comment_rating.append((comments[i],attr, one_hot_rating))

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

def process_csv(csv_path,ratings_len):
    pd_ratings = pd.read_csv(csv_path)
    comments = pd_ratings.comment[0:ratings_len]
    rating =to_categorical( pd_ratings.rating[0:ratings_len])
    rating_env =to_categorical( pd_ratings.rating_env[0:ratings_len])
    rating_flavor =to_categorical( pd_ratings.rating_flavor[0:ratings_len])
    rating_service = to_categorical(pd_ratings.rating_service[0:ratings_len])
    return comments,rating_env,rating_flavor,rating_service,rating

comment_rating=process_csv_to_data(data_path+'train40000.csv',ratings_len)
random_order = range(len(comment_rating))
np.random.shuffle(comment_rating)
np.random.shuffle(comment_rating)
np.random.shuffle(comment_rating)
train_data = [comment_rating[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [comment_rating[j] for i, j in enumerate(random_order) if i % 10 == 0]

from keras.preprocessing.text import Tokenizer
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = np.zeros([vec_space.shape[1]])
        for c in text:
            if c in unichars:
                R +=vec_space[ciDict.get(y)]
        return R

tokenizer = OurTokenizer(unichars)

def generator(data,min_index=0,max_index=None,
              shuffle=False,batch_size=16):
    if max_index is None:
        max_index=len(data)-1
    i= min_index
    while 1:
        if shuffle:
            rows = np.random.randint(
                max_index,size=batch_size)
        else:
            if i+batch_size>max_index:
                i = min_index
            rows = np.arange(i,min(i+batch_size,max_index))
            i+=batch_size
        X1, X2, Y = [], [], []
        for row in rows:
            X1.append(tokenizer._tokenize(data[row][0]))
            X2.append(tokenizer._tokenize(data[row][1]))
            Y.append(data[row][2])
        yield [np.array(X1), np.array(X2)], np.array(Y)

train_D = generator(train_data,batch_size=batch_size)
valid_D = generator(valid_data,batch_size=batch_size)
train_steps = len(train_data) //batch_size
val_steps =len(valid_data) //batch_size
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

history = model.fit_generator(train_D,
                              steps_per_epoch=train_steps,
                              epochs=20,
                              validation_data=valid_D,
                              validation_steps=val_steps,
                              callbacks=callbacks_list)
test_comments,test_rating_env,test_rating_flavor,test_rating_service,_=process_csv(data_path+'valid40000.csv',1000)
def load_data(test_comments,test_rating_env,test_rating_flavor,test_rating_service):
    test_in1 =[]
    test_in2=[]
    test_out=[]
    for i in range(len(test_comments)):
        test_in1.append(tokenizer._tokenize(test_comments[i]))
        test_in2.append(tokenizer._tokenize('口味'))
        test_out.append(test_rating_flavor)
        test_in1.append(tokenizer._tokenize(test_comments[i]))
        test_in2.append(tokenizer._tokenize('环境'))
        test_out.append(test_rating_env)
        test_in1.append(tokenizer._tokenize(test_comments[i]))
        test_in2.append(tokenizer._tokenize('服务'))
        test_out.append(test_rating_service)
    return np.array(test_in1),np.array(test_in2).np.array(test_out)

test_in1,test_in2,test_out=load_data(test_comments,test_rating_env,test_rating_flavor,test_rating_service)
results = model.evaluate([test_in1,test_in2],test_out)
print(results)

predictions = model.predict([test_in1,test_in2])
pred_out = np.argmax(predictions, axis=1)
y_out = np.argmax(test_out, axis=1)
print(pred_out[:100])
print(y_out[:100])
np.savetxt(os.path.join(save_file_path,time_id+'pred.txt'), pred_out, fmt = "%d", delimiter = ",")
np.savetxt(os.path.join(save_file_path,time_id+'act.txt'), y_out, fmt = "%d", delimiter = ",")
