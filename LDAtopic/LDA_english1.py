import gensim
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer

stemmer = PorterStemmer()
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

test1='This disk has failed many times. I would like to get it replaced.'
preprocess(test1)

def load_data(path):
    data = u''
    with open(path,'r') as f:
        for line in f:
            val =line.replace('\n','')
            data+=val
    return data

docs=load_data('/home/zju/buwenfeng/paperDemos/data/SemEval/2014task4/train_topic4.txt')
processed_docs=preprocess(docs)
dictionary = gensim.corpora.Dictionary([processed_docs])

bow_corpus = [dictionary.doc2bow([doc]) for doc in processed_docs]
lda_model =  gensim.models.LdaMulticore(bow_corpus,
                                   num_topics = 1,
                                   id2word = dictionary,
                                   passes = 10,
                                   workers = 2)

# lda_model.print_topic(0,topn=20)
lda_model.show_topic(0,topn=20)
