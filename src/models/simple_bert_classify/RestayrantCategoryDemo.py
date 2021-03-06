import xml.etree.cElementTree as ET
import numpy as np
dataset = 'Restaurants'
train_or_test ='train'
save_file_path ='/home/zju/buwenfeng/autosave'
load_model_path ='/home/zju/buwenfeng/autosave/trainSev2014Restaurants1553/my_model.h5'

train_epochs =4
# Parse Training File
def process_data(path):
    train_sentences = []
    tree = ET.ElementTree(file=path)
    for index, sentence in enumerate(tree.iter(tag='sentence')):
        for elem in sentence.iter():
            if (elem.tag == 'text'):
                text = elem.text
            elif (elem.tag == 'aspectCategories'):
                for at in elem.iter():
                    if ('category' not in at.attrib):
                        continue
                    s = {}
                    s['text'] = []
                    s['text'] = text
                    s['category'] = []
                    s['polarity'] = []
                    s['category'].append(at.attrib['category'])
                    s['polarity'].append(at.attrib['polarity'])
                    train_sentences.append(s)
    return train_sentences

train_sentences =process_data('/home/zju/buwenfeng/paperDemos/data/SemEval/2014task4/{}_Train.xml'.format(dataset))
test_sentences =process_data('/home/zju/buwenfeng/paperDemos/data/SemEval/2014task4/{}_Test.xml'.format(dataset))



import codecs

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

maxlen=170
tokenizer = Tokenizer(token_dict)

from keras_preprocessing import sequence
pad_sequences = sequence.pad_sequences
from keras.utils import to_categorical
def generate_data(train_sentences):
    x_train1 = []
    x_train2 = []
    y_train = []
    y_val=0
    for s in train_sentences:
        indices, segments = tokenizer.encode(first=s['text'][0][:100], second=s['aspectTerm'][0], max_len=maxlen)
        x_train1.append(indices)
        x_train2.append(segments)
        if s['polarity'][0]=='neutral':
            y_val=0
        elif s['polarity'][0] == 'negative':
            y_val=1
        elif s['polarity'][0] == 'positive':
            y_val=2
        elif s['polarity'][0] == 'conflict':
            y_val=3
        else:
            y_val=4
        y_train.append(y_val)
    y_train = to_categorical(y_train)
    y_data =[]
    for score in y_train:
        y_data.append([score])
    return [x_train1,x_train2],np.array(y_data)

x_train,y_train=generate_data(train_sentences)
x_test,y_test=generate_data(test_sentences)

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
x = bert_model([x1_in, x2_in])
x=Dense(128,activation='relu')(x)
x=Dropout(0.5)(x) # Dropout正则化
x =Dense(32,kernel_regularizer=regularizers.l2(0.001),activation='relu')(x)
x=Dropout(0.5)(x)
x =Dense(16,activation='relu')(x)
x=Dropout(0.5)(x)
p = Dense(4, activation='softmax')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-4),  # 因为参数少了，增大学习率
    metrics=['accuracy'])

import datetime,os
time_id = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute)
import keras
def run_model(train_or_test ='train',model_path =None):
    if train_or_test =='train':
        save_path = '/home/zju/buwenfeng/autosave/trainSev2014' + dataset + time_id + "/";
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='acc',
                patience=1,  # 大于1轮，也就是两轮没进步就停下来
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=save_path + 'my_model.h5',
                monitor='val_loss',  # val loss 没有改变就不需要覆盖模型
                save_best_only=True,
            )]
        history = model.fit(x_train, y_train,
                            epochs=train_epochs,
                            batch_size=16,
                            validation_split=0.2,
                            callbacks=callbacks_list)
    elif train_or_test =='test':
        model.load_weights(model_path)
    return model

run_model(train_or_test,load_model_path)
results = model.evaluate(x_test,y_test)
print(results)
if train_or_test =='test':
    predictions = model.predict(x_test)
    pred_out = np.sum(predictions, axis=1)
    pred_out = np.argmax(pred_out, axis=1)
    y_out = np.argmax(y_test, axis=2)
    y_out =np.transpose(y_out).squeeze()
    print(pred_out[:100])
    print(y_out[:100])
    np.savetxt(os.path.join(save_file_path,time_id+'pred.txt'), pred_out, fmt = "%d", delimiter = ",")
    np.savetxt(os.path.join(save_file_path,time_id+'act.txt'), y_out, fmt = "%d", delimiter = ",")