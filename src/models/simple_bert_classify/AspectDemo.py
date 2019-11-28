#!/usr/bin/env python
# Script to prepare datasets
import xml.etree.cElementTree as ET
import numpy as np
import string
import os
import re

dataset = 'Laptop'
mode = 'term'
train_sentences = []
test_sentences = []
aspectTerms = []
aspectCats = []
words = []
toy = False
one_hot = False

def strip_punctuation(s):
    return s.translate(s.maketrans('',''))


# return ''.join(c for c in s if c not in punctuation)

def process_text(x):
    x = x.lower()
    x = re.sub('[^A-Za-z0-9]+', ' ', x)
    x = x.split(' ')
    x = [strip_punctuation(y) for y in x]
    return x


# Parse Training File 
tree = ET.ElementTree(file='/home/zju/buwenfeng/paperDemos/data/SemEval/2014task4/{}_Train.xml'.format(dataset))
for index, sentence in enumerate(tree.iter(tag='sentence')):
    s = {}
    for elem in sentence.iter():
        if (elem.tag == 'text'):
            ptxt = process_text(elem.text)
            s['text'] = ptxt
            words += ptxt
        elif (elem.tag == 'aspectTerms' and mode == 'term'):
            s['aspectTerms'] = []
            for at in elem.iter():
                attr = at.attrib
                if ('term' not in attr):
                    continue
                txt = process_text(at.attrib['term'])
                print(txt)
                words += txt
                s['aspectTerms'].append([txt, at.attrib])
                aspectTerms.append(txt)
        elif (elem.tag == 'aspectCategories' and mode == 'aspect'):
            s['aspectCats'] = []
            for ac in elem.iter():
                attr = ac.attrib
                if ('category' not in attr):
                    continue
                s['aspectCats'].append([attr['category'], attr])
            aspectCats.append(attr['category'])
    train_sentences.append(s)

all_text = []

# Parse Testing File
tree = ET.ElementTree(file='/home/zju/buwenfeng/paperDemos/data/SemEval/2014task4/{}_Test.xml'.format(dataset))
for index, sentence in enumerate(tree.iter(tag='sentence')):
    s = {}
    for elem in sentence.iter():
        if (elem.tag == 'text'):
            ptxt = process_text(elem.text)
            all_text.append(ptxt)
            s['text'] = ptxt
            words += ptxt
        elif (elem.tag == 'aspectTerms' and mode == 'term'):
            s['aspectTerms'] = []
            for at in elem.iter():
                attr = at.attrib
                if ('term' not in attr):
                    continue
                txt = process_text(at.attrib['term'])
                print(txt)
                words += txt
                s['aspectTerms'].append([txt, at.attrib])
                aspectTerms.append(txt)
        elif (elem.tag == 'aspectCategories' and mode == 'aspect'):
            s['aspectCats'] = []
            for ac in elem.iter():
                attr = ac.attrib
                if ('category' not in attr):
                    continue
                s['aspectCats'].append([attr['category'], attr])
    test_sentences.append(s)

#aspectTerms = list(set(aspectTerms))
aspectCats = list(set(aspectCats))
if (mode == 'term'):
    term_lens = [len(x) for x in aspectTerms]

# words += aspectTerms 
words += aspectCats
words = list(set(words))
all_lens = [len(x) for x in all_text]
max_len, avg_len, min_len = np.max(all_lens), np.mean(all_lens), np.min(all_lens)
if (mode == 'term'):
    max_term_len = np.max(term_lens)
    print("{} aspect terms".format(len(aspectTerms)))
    print("{} max len for aspect terms".format(np.max(term_lens)))
#3012 aspect terms
#8 max len for aspect terms
if (mode == 'aspect'):
    print("{} aspect categories".format(len(aspectCats)))
print("{} unique words".format(len(words)))
print("{} train sentences".format(len(train_sentences)))#3045 train sentences
print("{} test sentences".format(len(test_sentences)))
print("max sent={} avg sent={} min sent={}".format(max_len, avg_len, min_len))

# Building vocab indices
index_word = {index + 2: word for index, word in enumerate(words)}
word_index = {word: index + 2 for index, word in enumerate(words)}
index_word[0], index_word[1] = '<pad>', '<unk>'
word_index['<pad>'], word_index['unk'] = 0, 1
if (mode == 'aspect'):
    index_cat = {index: word for index, word in enumerate(aspectCats)}
    cat_index = {word: index for index, word in enumerate(aspectCats)}











