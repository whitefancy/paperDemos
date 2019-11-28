#对不同的主题提扩充更多的主题词
data_path='../data/NLPCC2016_Stance_Detection_Test_Datasets/evasampledata4-TaskAA.txt'
def process_txt_file(path):
    target = []
    text = []
    stance = []
    with open(path) as f:
        for line in f:
            values = line.split('\t')
            if len(values) == 4:
                target.append(values[1])
                text.append(values[2])
                stance.append(values[3])
    return target,text,stance
train_targets,train_texts,train_stances=process_txt_file(data_path)
import numpy as np
import jieba
topics = list(set(train_targets))
LDA_data =[]
for i in range(len(topics)):
    tt=[ x for ix,x in enumerate(train_texts) if train_targets[ix] == topics[i]]
    LDA_data.append(tt)
    for document in tt:
        document_cut = jieba.cut(document)
        # document_cut = jieba.cut_for_search(document)
        result = ' '.join(document_cut)
        with open('./topic'+str(i)+'.txt', 'a') as f2:
            f2.write(result)

# 从文件导入停用词表
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
# 将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
cntVector = CountVectorizer(stop_words=stpwrdlst)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

for i in range(len(topics)):
    with open('./topic' + str(i) + '.txt', 'r') as f2:
        res2 = f2.read()
        cntTf = cntVector.fit_transform([res2])
        lda = LatentDirichletAllocation(n_topics=1,
                                        learning_offset=50.,
                                        random_state=0)
        docres = lda.fit_transform(cntTf)
        tf_feature_names = cntVector.get_feature_names()
        print_top_words(lda, tf_feature_names,50)
#搜索模式分词
#俄罗斯 叙利亚 土耳其 美国 罗斯 反恐 利亚 普京 恐怖 恐怖分子
#iphone se 手机 苹果 5s 6s 屏幕 iphonese 小屏 价格
#春节 放鞭炮 鞭炮 放鞭 传统 环保 过年 我们 烟花 cn
#深圳 禁摩 限电 电动车 电动 动车 交警 http cn 自行车
#二胎 开放 政策 孩子 一个 国家 现在 我们 自己 没有

#精确模式分词
#二胎 开放 政策 孩子 一个 国家 我们 现在 自己 没有
#深圳 禁摩 限电 电动车 电动 交警 http cn 动车 自行车
#春节 放鞭炮 鞭炮 放鞭 传统 环保 过年 我们 cn http
#iphone se 手机 苹果 5s 6s 屏幕 iphonese 小屏 价格
#俄罗斯 叙利亚 土耳其 美国 反恐 普京 恐怖分子 就是 罗斯 中国

# 筛选后