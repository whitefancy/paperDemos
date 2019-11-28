from sklearn.metrics.pairwise import cosine_similarity
from keras_bert import extract_embeddings

model_path = '/home/zju/buwenfeng/bert/chinese_L-12_H-768_A-12/'
texts = LDA_data[3]
topics =[
'支持苹果手机se喜欢',
'外观好看屏幕小合适',
'配置内存性能很满意',
'价格便宜市场发布不错',
'喜欢系统功能设计良好' ]
topics1 =['深圳禁摩',
'电动车自行车',
'交警执法',
'快递外卖',
'摩托车电摩三轮车',
'整治非法拘留' ]
import numpy as np
text_embeddings = extract_embeddings(model_path, texts)
topic_embeddings = extract_embeddings(model_path, topics)
texteb1 = extract_embeddings(model_path, [LDA_data[2][20]])
for ix,x in enumerate(text_embeddings):
    print(texts[ix]+'\n')
    if ix==40:
        break
    for it,t in enumerate(topic_embeddings):
        same_bank = np.average(np.matmul(x,np.transpose(t)))
        print(topics[it] + ' '+str(same_bank)+'\n')
