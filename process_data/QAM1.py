data_path='/home/zju/buwenfeng/TensorFlowDemo/data/NLPCC2016_Stance_Detection_Test_Datasets'
import numpy as np
train_or_test ='train'
maxlen=400
save_file_path ='/home/zju/buwenfeng/autosave'
load_model_path ='/home/zju/buwenfeng/autosave/stenceweibo1653/'
bert_path='/home/zju/buwenfeng/bert/chinese_L-12_H-768_A-12/'
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
relative_path=bert_path
config_path =relative_path+ 'bert_config.json'
checkpoint_path = relative_path+ 'bert_model.ckpt'
dict_path =relative_path+'vocab.txt'
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)
train_epochs =10
mode = 'QA-B'
topics=['IphoneSE','春节放鞭炮','俄罗斯在叙利亚的反恐行动','开放二胎','深圳禁摩限电']
LDA_topics=['IphoneSE','春节放鞭炮','俄罗斯在叙利亚的反恐行动','开放二胎','深圳禁摩限电']
topicDict = dict((ch,ind)for ch,ind in zip(topics,LDA_topics))
def load_data(path):
    data=[]
    with open(path,'r') as f:
        for line in f:
            val =line.replace('\n','')
            data.append(val)
    return data

import os
train_topic = load_data(os.path.join(data_path,'train_topic.txt'))
test_topic =load_data(os.path.join(data_path,'test_topic.txt'))
train_text =load_data(os.path.join(data_path,'train_text.txt'))
test_text = load_data(os.path.join(data_path,'test_text.txt'))
train_stance = load_data(os.path.join(data_path,'train_stance.txt'))
test_stance = load_data(os.path.join(data_path,'test_stance.txt'))
QAB_st=['你怎么看待','？']
from keras.utils import to_categorical
def genete_data1(topic,texts,stances,mode = 'QA-B'):
    AGAINST = 0
    FAVOR = 1
    NONE =2
    x1=[]
    x2=[]
    target=[]
    ids=[]
    for ix,x in enumerate(texts):
        top_ix =topicDict[topic[ix]]
        if mode == 'QA-B':
            indices, segments = tokenizer.encode(first=x, second=QAB_st[0] + top_ix+QAB_st[1], max_len=maxlen)
            x1.append(indices)
            x2.append(segments)
            ids.append(ix)
            if stances[ix] == 'AGAINST':
                target.append(AGAINST)
            elif stances[ix] == 'FAVOR':
                target.append(FAVOR)
            elif stances[ix] == 'NONE':
                target.append(NONE)
            else:
                target.append(NONE)
    target1 = to_categorical(target)
    return x1,x2,target1,ids,target

train_x1,train_x2,train_target,_ ,_= genete_data1(train_topic,train_text,train_stance)
test_x1,test_x2,test_target,_,test_tar = genete_data1(test_topic,test_text,test_stance)

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(maxlen,))
x2_in = Input(shape=(maxlen,))
x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(3, activation='softmax')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['accuracy'])

import datetime,os
time_id = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute)
save_path = '/home/zju/buwenfeng/autosave/stenceweibo' + time_id + "/"
if train_or_test =='test':
    save_path=load_model_path
import keras
def run_model(train_or_test ='train',model_path =None):
    if train_or_test =='train':
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='accuracy',
                patience=1,  # 大于1轮，也就是两轮没进步就停下来
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=save_path + 'my_model.h5',
                monitor='val_loss',  # val loss 没有降低就不需要覆盖模型
                save_best_only=True,
            )]
        history = model.fit([train_x1,train_x2], np.array(train_target),
                            epochs=train_epochs,
                            batch_size=4,
                            validation_split=0.2,
                            callbacks=callbacks_list)

run_model(train_or_test,load_model_path)

results = model.evaluate([test_x1,test_x2],np.array(test_target))
print(results)
predictions = model.predict([test_x1,test_x2])
pred_out = np.argmax(predictions,axis=1)
np.savetxt(os.path.join(save_file_path,time_id+'pred.txt'), pred_out, fmt = "%d", delimiter = ",")
np.savetxt(os.path.join(save_file_path,time_id+'act.txt'), test_tar, fmt = "%d", delimiter = ",")
from sklearn.metrics import classification_report
target_names =['against','favor','none']
print(classification_report(test_tar,pred_out,target_names=target_names))

model.load_weights(save_path + 'my_model.h5')
results = model.evaluate([test_x1,test_x2],np.array(test_target))
print(results)
predictions = model.predict([test_x1,test_x2])
pred_out = np.argmax(predictions,axis=1)
from sklearn.metrics import classification_report
target_names =['against','favor','none']
print(classification_report(test_tar,pred_out,target_names=target_names))
