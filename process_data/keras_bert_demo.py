from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
maxlen = 500# 在500个单词之后截断文本，同时这些单词都属于前max_features个最常见的单词
relative_path='/home/zju/buwenfeng/bert/multi_cased_L-12_H-768_A-12/';
config_path =relative_path+ 'bert_config.json'
checkpoint_path = relative_path+ 'bert_model.ckpt'
dict_path =relative_path+'vocab.txt'


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

from keras_bert import Tokenizer

token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,
    'un': 2,
    '##aff': 3,
    '##able': 4,
    '[UNK]': 5,
}
tokenizer = Tokenizer(token_dict)
print(tokenizer.tokenize('unaffable'))  # The result should be `['[CLS]', 'un', '##aff', '##able', '[SEP]']`
indices, segments = tokenizer.encode('unaffable')
print(indices)  # Should be `[0, 2, 3, 4, 1]`
print(segments)  # Should be `[0, 0, 0, 0, 0]`

print(tokenizer.tokenize(first='unaffable', second='钢'))
# The result should be `['[CLS]', 'un', '##aff', '##able', '[SEP]', '钢', '[SEP]']`
indices, segments = tokenizer.encode(first='unaffable', second='钢', max_len=10)
print(indices)  # Should be `[0, 2, 3, 4, 1, 5, 1, 0, 0, 0]`
print(segments)  # Should be `[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]`

#训练预模型
import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs


# A toy input example
sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
]


# Build token dictionary
token_dict = get_base_dict()  # A dict that contains some special tokens
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word


# Build & train the model
model = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
compile_model(model)
model.summary()

def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )

model.fit_generator(
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=100,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)


# Use the trained model
inputs, output_layer = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,      # The input layers and output layer will be returned if `training` is `False`
    trainable=False,     # Whether the model is trainable. The default value is the same with `training`
    output_layer_num=4,  # The number of layers whose outputs will be concatenated as a single output.
                         # Only available when `training` is `False`.
)

#使用带预热的学习率
import numpy as np
from keras_bert import AdamWarmup, calc_train_steps

train_x = np.random.standard_normal((1024, 100))

total_steps, warmup_steps = calc_train_steps(
    num_example=train_x.shape[0],
    batch_size=32,
    epochs=10,
    warmup_proportion=0.1,
)

optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)

#在线获取预训练的检查点
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths

model_path = get_pretrained(PretrainedList.multi_cased_base)
paths = get_checkpoint_paths(model_path)
print(paths.config, paths.checkpoint, paths.vocab)


#使用bert对单个句子提取特征
from keras_bert import extract_embeddings

model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'
texts = ['all work and no play', 'makes jack a dull boy~']

embeddings = extract_embeddings(model_path, texts)

#对成对句子提取特征 需要取出最后4层
from keras_bert import extract_embeddings, POOL_NSP, POOL_MAX
model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'
texts = [
    ('all work and no play', 'makes jack a dull boy'),
    ('makes jack a dull boy', 'all work and no play'),
]
embeddings = extract_embeddings(model_path, texts, output_layer_num=4, poolings=[POOL_NSP, POOL_MAX])

#对文本文件提取特征
import codecs
from keras_bert import extract_embeddings
model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'
with codecs.open('xxx.txt', 'r', 'utf8') as reader:
    texts = map(lambda x: x.strip(), reader)
    embeddings = extract_embeddings(model_path, texts)

#进行微调

import os
from keras_bert import load_trained_model_from_checkpoint

layer_num = 12
checkpoint_path = '.../uncased_L-12_H-768_A-12'

config_path = os.path.join(checkpoint_path, 'bert_config.json')
model_path = os.path.join(checkpoint_path, 'bert_model.ckpt')
model = load_trained_model_from_checkpoint(
    config_path,
    model_path,
    training=False,
    use_adapter=True,
    trainable=['Encoder-{}-MultiHeadSelfAttention-Adapter'.format(i + 1) for i in range(layer_num)] +
    ['Encoder-{}-FeedForward-Adapter'.format(i + 1) for i in range(layer_num)] +
    ['Encoder-{}-MultiHeadSelfAttention-Norm'.format(i + 1) for i in range(layer_num)] +
    ['Encoder-{}-FeedForward-Norm'.format(i + 1) for i in range(layer_num)],
)

#使用TensorFlow内置的keras后端
TF_KERAS=1
#使用Theano的后端
KERAS_BACKEND=theano
