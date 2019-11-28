pred_path='/home/zju/buwenfeng/autosave/1457pred.txt'
act_path='/home/zju/buwenfeng/autosave/1457act.txt'
id_path='/home/zju/buwenfeng/autosave/1457test_id.txt'

preds=[]
acts=[]
ids=[]
with open(pred_path,'r') as f:
    for line in f:
        val =line.replace('\n','')
        preds.append(int(val))

with open(act_path,'r') as f:
    for line in f:
        val =line.replace('\n','')
        acts.append(int(val))

with open(id_path,'r') as f:
    for line in f:
        val =line.replace('\n','')
        ids.append(int(val))

fpred=[]
fact=[]
import numpy as np
for ix,x in enumerate(ids):
    if(ix%3==0):
        pred_sum=np.sum(preds[ix:ix+3])
        if(pred_sum==2):
            pred_ix=[ip for ip,p in enumerate(preds[ix:ix+3]) if p==0][0]
        elif(pred_sum==3):
            pred_ix=2
        elif(pred_sum==1):
            pred_ix = [ip for ip,p in enumerate(preds[ix:ix+3]) if p==1][0]
        else:
            pred_ix = 2
        act_ix =[ip for ip,p in enumerate(acts[ix:ix+3]) if p==0][0]
        fpred.append(pred_ix)
        fact.append(act_ix)

from sklearn.metrics import classification_report
target_names =['against','favor','none']
print(classification_report(fact,fpred,target_names=target_names))