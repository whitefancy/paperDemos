pred_path='/home/zju/buwenfeng/autosave/1653pred.txt'
act_path='/home/zju/buwenfeng/autosave/1653act.txt'

preds=[]
acts=[]
with open(pred_path,'r') as f:
    for line in f:
        val =line.replace('\n','')
        preds.append(int(val))

with open(act_path,'r') as f:
    for line in f:
        val =line.replace('\n','')
        acts.append(int(val))

import numpy as np

from sklearn.metrics import classification_report
target_names =['against','favor','none']
print(classification_report(acts,preds,target_names=target_names))