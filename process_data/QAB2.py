data_path='/home/zju/buwenfeng/TensorFlowDemo/data/NLPCC2016_Stance_Detection_Test_Datasets'
import numpy as np
train_or_test ='train'
maxlen=400
save_file_path ='/home/zju/buwenfeng/autosave'
load_model_path ='/home/zju/buwenfeng/autosave/stenceweibo1630/my_model.h5'
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
LDA_topics=['iphonese手机苹果5s6s屏幕iphonese小屏价格没有还是一个喜欢就是可以配置外观但是这个现在一样市场感觉觉得英寸发布中国不是真的大屏使用什么产品不会性能因为怎么内存便宜已经系统设计功能那么认为很多知道用户不错',
'春节放鞭炮鞭炮放鞭传统环保过年我们cnhttp烟花中国爆竹大家今年污染燃放空气烟花爆竹一个今年春节节日可以期间为了环境没有今天燃放烟花习俗这个新年市民什么回家春联呼吁怎么文化倡议传统节日就是安全民俗还是少放自己减少孩子放炮',
'俄罗斯叙利亚土耳其美国反恐普京恐怖分子就是罗斯中国is支持国家利亚恐怖世界反对派isis没有利益自己不是一个他们分子中东组织毛子打击轰炸伊斯兰问题恐怖组织什么表示现在你们行动反对北约政府沙特应该不会总统可能知道亲人飞机大国',
'二胎开放政策孩子一个国家我们现在自己没有什么生二胎可以生育就是时候问题这个中国女性怀孕还是已经人口不是那么这么结婚妈妈家庭真的宝宝工作你们觉得姚晨如果为什么怎么女人今天知道计划生育以后不能儿子父母但是一起不要',
'深圳禁摩限电电动车电动交警httpcn动车自行车快递摩托车政府城市没有你们汽车不是交通电摩什么行动应该整治管理可以就是中国摩托他们我们因为不能拘留外卖非法执法问题支持三轮车这样怎么国家政策一个自行百姓机动车为什么行车'
]
topicDict = dict((ch,ind)for ch,ind in zip(topics,LDA_topics))
def load_data(path):
    data=[]
    with open(path,'r') as f:
        for line in f:
            val =line.replace('\n','')
            data.append(val)
    return data

import os
train_topic = load_data(os.path.join(data_path,'train_topic0.txt'))
test_topic =load_data(os.path.join(data_path,'test_topic0.txt'))
train_text =load_data(os.path.join(data_path,'train_text0.txt'))
test_text = load_data(os.path.join(data_path,'test_text0.txt'))
train_stance = load_data(os.path.join(data_path,'train_stance0.txt'))
test_stance = load_data(os.path.join(data_path,'test_stance0.txt'))
QAB_st=['我反对','我支持','我不知道']

def genete_data1(topic,texts,stances,mode = 'QA-B'):
    Yes = 0
    No = 1
    x1=[]
    x2=[]
    target=[]
    ids=[]
    for ix,x in enumerate(texts):
        top_ix =topicDict[topic[ix]]
        if mode == 'QA-B':  # 我反对
            for senti_st in QAB_st:
                indices, segments = tokenizer.encode(first=senti_st + top_ix, second=x, max_len=maxlen)
                x1.append(indices)
                x2.append(segments)
                ids.append(ix)
            if stances[ix] == 'AGAINST':
                target.append(Yes)
                target.append(No)
                target.append(No)
            elif stances[ix] == 'FAVOR':
                target.append(No)
                target.append(Yes)
                target.append(No)
            elif stances[ix] == 'NONE':
                target.append(No)
                target.append(No)
                target.append(Yes)
            else:
                target.append(No)
                target.append(No)
                target.append(No)
    return x1,x2,target,ids

train_x1,train_x2,train_target,_ = genete_data1(train_topic,train_text,train_stance)
test_x1,test_x2,test_target,test_id = genete_data1(test_topic,test_text,test_stance)

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
p = Dense(1, activation='sigmoid')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['accuracy'])

import datetime,os
time_id = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute)
save_path = '/home/zju/buwenfeng/autosave/stenceweibo' + time_id + "/"
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
                monitor='val_loss',  # val loss 没有改变就不需要覆盖模型
                save_best_only=True,
            )]
        history = model.fit([train_x1,train_x2], np.array(train_target),
                            epochs=train_epochs,
                            batch_size=4,
                            validation_split=0.2,
                            callbacks=callbacks_list)
    elif train_or_test =='test':
        model.load_weights(model_path)

run_model(train_or_test,load_model_path)


results = model.evaluate([test_x1,test_x2],np.array(test_target))
print(results)

predictions = model.predict([test_x1,test_x2])
pred_out = [1if p>0.5 else 0 for p in predictions]

fpred=[]
fact=[]
import numpy as np
for ix,x in enumerate(test_id):
    if(ix%3==0):
        pred_sum=np.sum(pred_out[ix:ix+3])
        if(pred_sum==2):
            pred_ix=[ip for ip,p in enumerate(pred_out[ix:ix+3]) if p==0][0]
        elif(pred_sum==3):
            pred_ix=2
        elif(pred_sum==1):
            pred_ix = [ip for ip,p in enumerate(pred_out[ix:ix+3]) if p==1][0]
        else:
            pred_ix = 2
        act_ix =[ip for ip,p in enumerate(test_target[ix:ix+3]) if p==0][0]
        fpred.append(pred_ix)
        fact.append(act_ix)

from sklearn.metrics import classification_report
target_names =['against','favor','none']
print(classification_report(fact,fpred,target_names=target_names))

if train_or_test =='train':
    model.load_weights(save_path + 'my_model.h5')
    results = model.evaluate([test_x1,test_x2],np.array(test_target))
    print(results)
    predictions = model.predict([test_x1,test_x2])
    pred_out = [1if p>0.5 else 0 for p in predictions]
    np.savetxt(os.path.join(save_file_path,time_id+'pred.txt'), pred_out, fmt = "%d", delimiter = ",")
    np.savetxt(os.path.join(save_file_path,time_id+'act.txt'), test_target, fmt = "%d", delimiter = ",")
    np.savetxt(os.path.join(save_file_path,time_id+'test_id.txt'), test_id, fmt = "%d", delimiter = ",")
    fpred=[]
    fact=[]
    import numpy as np
    for ix,x in enumerate(test_id):
        if(ix%3==0):
            pred_sum=np.sum(pred_out[ix:ix+3])
            if(pred_sum==2):
                pred_ix=[ip for ip,p in enumerate(pred_out[ix:ix+3]) if p==0][0]
            elif(pred_sum==3):
                pred_ix=2
            elif(pred_sum==1):
                pred_ix = [ip for ip,p in enumerate(pred_out[ix:ix+3]) if p==1][0]
            else:
                pred_ix = 2
            act_ix =[ip for ip,p in enumerate(test_target[ix:ix+3]) if p==0][0]
            fpred.append(pred_ix)
            fact.append(act_ix)
    from sklearn.metrics import classification_report
    target_names =['against','favor','none']
    print(classification_report(fact,fpred,target_names=target_names))
