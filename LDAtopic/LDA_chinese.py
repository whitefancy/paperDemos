# -*- coding: utf-8 -*-

import jieba

jieba.suggest_freq('沙瑞金', True)#使用 suggest_freq(segment, tune=True) 可调节单个词语的词频，使其能（或不能）被分出来。
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)
# 第一个文档分词#
with open('./nlp_test0.txt') as f:
    document = f.read()
    document_cut = jieba.cut(document)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document_cut)
    with open('./nlp_test1.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()

# 第二个文档分词#
with open('./nlp_test2.txt') as f:
    document2 = f.read()
    document2_cut = jieba.cut(document2)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document2_cut)
    with open('./nlp_test3.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()

# 第三个文档分词#
jieba.suggest_freq('桓温', True)
with open('./nlp_test4.txt') as f:
    document3 = f.read()
    document3_cut = jieba.cut(document3)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document3_cut)
    with open('./nlp_test5.txt', 'w') as f3:
        f3.write(result)
f.close()
f3.close()

with open('./nlp_test1.txt') as f3:
    res1 = f3.read()
print (res1)
with open('./nlp_test3.txt') as f4:
    res2 = f4.read()
print(res2)
with open('./nlp_test5.txt') as f5:
    res3 = f5.read()
print(res3)

#从文件导入停用词表
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
corpus = [res1,res2,res3]
cntVector = CountVectorizer(stop_words=stpwrdlst)
cntTf = cntVector.fit_transform(corpus)
print(cntTf)

lda = LatentDirichletAllocation(n_topics=20,
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(cntTf)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

n_top_words = 10
tf_feature_names = cntVector.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)