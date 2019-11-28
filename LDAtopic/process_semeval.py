import xml.etree.cElementTree as ET
import numpy as np
dataset = 'Restaurants'
save_file_path ='/home/zju/buwenfeng/paperDemos/data/SemEval/2014task4'
texts=[]
categorys=[]
def process_data(path):
    tree = ET.ElementTree(file=path)
    for index, sentence in enumerate(tree.iter(tag='sentence')):
        for elem in sentence.iter():
            if (elem.tag == 'text'):
                text = elem.text
            elif (elem.tag == 'aspectCategories'):
                for at in elem.iter():
                    if ('category' not in at.attrib):
                        continue
                    texts.append(text)
                    categorys.append(at.attrib['category'])

process_data('/home/zju/buwenfeng/paperDemos/data/SemEval/2014task4/{}_Train.xml'.format(dataset))
import os
cate=list(set(categorys))
np.savetxt(os.path.join(save_file_path,'train_topic'+dataset+'.txt'), cate, fmt = "%s", delimiter = " ")
for ix,x in enumerate(cate):
    texts1=[t for it,t in enumerate(texts) if categorys[it] ==x]
    np.savetxt(os.path.join(save_file_path, 'train_topic' + str(ix) + '.txt'), texts1, fmt="%s", delimiter=" ")

texts = []
categorys = []
process_data('/home/zju/buwenfeng/paperDemos/data/SemEval/2014task4/{}_Test.xml'.format(dataset))
cate=list(set(categorys))
np.savetxt(os.path.join(save_file_path,'test_topic'+dataset+'.txt'), cate, fmt = "%s", delimiter = " ")
for ix,x in enumerate(cate):
    texts1=[t for it,t in enumerate(texts) if categorys[it] ==x]
    np.savetxt(os.path.join(save_file_path, 'test_topic' + str(ix) + '.txt'), texts1, fmt="%s", delimiter=" ")
