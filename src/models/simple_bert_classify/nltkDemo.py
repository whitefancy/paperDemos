import nltk
#查看提供的已经标注好的文本
nltk.corpus.brown.tagged_words()
nltk.corpus.brown.tagged_words(tagset='universal')
#查看nltk资源路径
nltk.data.path
#加载离线nltk资源
from nltk import data
data.path.append(r'/home/zju/nltk_data')
#标注句子词性
text = nltk.word_tokenize('Great product, great price, great delivery, and great service.')
text = nltk.word_tokenize()
nltk.pos_tag(text,tagset='universal')
text = nltk.word_tokenize('The Mountain Lion OS is not hard to figure out if you are familiar with Microsoft Windows.')

#中文标注句子词性 jieba
import jieba.posseg as psg
text = psg.cut('里面有一些进口食品还是不错的，但个人感觉商品种类比较少，而且管理不是很灵活，退货比较麻烦。价格方面是比较适中的。')
for ele in text:
    print(ele.word)
    print(ele.flag)