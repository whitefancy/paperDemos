data_path='/home/zju/buwenfeng/TensorFlowDemo/data/NLPCC2016_Stance_Detection_Test_Datasets/evasampledata4-TaskAA.txt'
import numpy as np
train_or_test ='train'
maxlen=400
save_file_path ='/home/zju/buwenfeng/autosave'
load_model_path ='/home/zju/buwenfeng/autosave/trainSev2014Restaurants134/my_model.h5'
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
train_epochs =5
mode = 'QA-B'
topic = []
text = []
stance = []
topics=['IphoneSE','春节放鞭炮','俄罗斯在叙利亚的反恐行动','开放二胎','深圳禁摩限电']
LDA_topics=['IphoneSE',
            '春节放鞭炮',
            '俄罗斯在叙利亚的反恐行动',
            '开放二胎',
            '深圳禁摩限电'
            ]
topicDict = dict((ch,ind)for ch,ind in zip(topics,LDA_topics))

with open(data_path) as f:
    for line in f:
        values = line.split('\t')
        if len(values) == 4:
            topic.append(values[1])
            text.append(values[2])
            stance.append(values[3].replace('\n',''))

random_order = np.linspace(0,len(stance)-1,num=len(stance)-1,dtype=int)
np.random.shuffle(random_order)
train_topic = [topic[j] for i, j in enumerate(random_order) if i % 10 != 0]
test_topic = [topic[j] for i, j in enumerate(random_order) if i % 10 == 0]
train_text = [text[j] for i, j in enumerate(random_order) if i % 10 != 0]
test_text = [text[j] for i, j in enumerate(random_order) if i % 10 == 0]
train_stance = [stance[j] for i, j in enumerate(random_order) if i % 10 != 0]
test_stance = [stance[j] for i, j in enumerate(random_order) if i % 10 == 0]
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
                indices, segments = tokenizer.encode(first=x, second=senti_st + top_ix, max_len=maxlen)
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
from keras_bert import AdamWarmup, calc_train_steps
total_steps, warmup_steps = calc_train_steps(
    num_example=len(train_x1),
    batch_size=4,
    epochs=train_epochs,
    warmup_proportion=0.1,
)

optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
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
    optimizer=optimizer,
    metrics=['accuracy'])

import datetime,os
time_id = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute)
import keras
def run_model(train_or_test ='train',model_path =None):
    if train_or_test =='train':
        save_path = '/home/zju/buwenfeng/autosave/stenceweibo' + time_id + "/";
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
pred_out = [1 if p>0.5 else 0 for p in predictions]
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