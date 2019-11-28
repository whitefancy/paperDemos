import xml.etree.cElementTree as ET

dataset = 'Laptop'
train_or_test ='train'
save_file_path ='/home/zju/buwenfeng/autosave'
load_model_path ='/home/zju/buwenfeng/autosave/trainSev2014Restaurants134/my_model.h5'
# Parse Training File
def process_data(path):
    texts =[]
    aspectTerms =[]
    polarities =[]
    tree = ET.ElementTree(file=path)
    for index, sentence in enumerate(tree.iter(tag='sentence')):
        for elem in sentence.iter():
            if (elem.tag == 'text'):
                text = elem.text
            elif (elem.tag == 'aspectTerms'):
                for at in elem.iter():
                    if ('term' not in at.attrib):
                        continue
                    texts.append(text)
                    aspectTerms.append(at.attrib['term'])
                    if at.attrib['polarity'] == 'neutral':
                        y_val = 0
                    elif at.attrib['polarity'] == 'negative':
                        y_val = 1
                    elif at.attrib['polarity'] == 'positive':
                        y_val = 2
                    elif at.attrib['polarity'] == 'conflict':
                        y_val = 3
                    else:
                        y_val = 4
                    polarities.append(y_val)
    return texts,aspectTerms,polarities

train_texts,train_aspectTerms,train_polarities =process_data('/home/zju/buwenfeng/paperDemos/data/SemEval/2014task4/{}_Train.xml'.format(dataset))
test_texts,test_aspectTerms,test_polarities =process_data('/home/zju/buwenfeng/paperDemos/data/SemEval/2014task4/{}_Test.xml'.format(dataset))
maxlen=170

from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
maxlen = 100
train_samples = 200
validation_samples = 10000
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)
squences = tokenizer.texts_to_sequences(train_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.'%len(word_index))
data = pad_sequences(squences,maxlen=maxlen)
from keras.utils import to_categorical
labels = to_categorical(train_polarities)
import os
glove_dir = '/home/zju/buwenfeng/glove'
embeddings_index = {}
f = open(os.path.join(glove_dir,'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embeddings_index[word] = coefs
f.close()
# 准备嵌入矩阵 （max_words,embedding_dim)
embedding_dim = 100
embedding_matrix = np.zeros((max_words,embedding_dim))
for word,i in word_index.items():
    if i<max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector #    找不到的词，嵌入向量为0

from keras_preprocessing import sequence
pad_sequences = sequence.pad_sequences

from keras.layers import Embedding
embedding_layer = Embedding(1000,64)
max_features = 10000
# 在当前问题上学习词向量
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding
model = Sequential()
model.add(Embedding(10000,100,input_length=maxlen))
model.add(Flatten())# 三维词嵌入向量展平成二维（samples，maxlen*8）
model.add(Dense(32,activation='relu'))
model.add(Dense(4,activation='softmax'))#仅仅根据单词之间的关系，不考虑句子结构

#将预训练的词嵌入加载到Embedding层中
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False #冻结Embedding层 以免丢失保存的信息 否则会被随机初始化

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()

import keras
import datetime,os
time_id=str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
save_path='/home/zju/buwenfeng/autosave/trainSev2014'+time_id+"/";
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
history = model.fit(data,labels,
                    epochs=60,
                    batch_size=64,
                    validation_split=0.2,
                    callbacks=callbacks_list)

squences = tokenizer.texts_to_sequences(test_texts)
x_test = pad_sequences(squences, maxlen=maxlen)
y_test=to_categorical(test_polarities)

results = model.evaluate(x_test,y_test)
print(results)
predictions = model.predict(x_test)
pred_out = np.argmax(predictions, axis=1)
y_out = np.argmax(y_test, axis=1)
print(pred_out[:100])
print(y_out[:100])
np.savetxt(os.path.join(save_file_path,time_id+'pred.txt'), pred_out, fmt = "%d", delimiter = ",")
np.savetxt(os.path.join(save_file_path,time_id+'act.txt'), y_out, fmt = "%d", delimiter = ",")
