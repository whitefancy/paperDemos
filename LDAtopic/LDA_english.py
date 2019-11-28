import numpy as np
import lda
import lda.datasets

titles = lda.datasets.load_reuters_titles()

for i in range(395):
    print(titles[i])

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()

model = lda.LDA(n_topics=5,n_iter=1500,random_state=1)
model.fit(X)
topic_word = model.topic_word_
n_top_words = 3

for i,topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    #输出每个主题所包含的单词的分布
    print('Topic{}:{}'.format(i,''.join(topic_words)))

doc_topic = model.doc_topic_
for i in range(20):
    #输出文章所对应的主题
    print("{} (top topic:{})".format(titles[i],doc_topic[i].argmax()))

