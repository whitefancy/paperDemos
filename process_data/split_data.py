data_path='/home/zju/buwenfeng/TensorFlowDemo/data/NLPCC2016_Stance_Detection_Test_Datasets/evasampledata4-TaskAA.txt'
save_file_path='/home/zju/buwenfeng/TensorFlowDemo/data/NLPCC2016_Stance_Detection_Test_Datasets'
topic = []
text = []
stance = []
import numpy as np
import os
with open(data_path) as f:
    for line in f:
        values = line.split('\t')
        if len(values) == 4:
            topic.append(values[1])
            text.append(values[2])
            stance.append(values[3].replace('\n',''))

random_order = np.linspace(0,len(stance)-1,num=len(stance)-1,dtype=int)
np.random.shuffle(random_order)
train_topic = [topic[j] for i, j in enumerate(random_order) if i % 10 != 0]
test_topic = [topic[j] for i, j in enumerate(random_order) if i % 10 == 0]
train_text = [text[j] for i, j in enumerate(random_order) if i % 10 != 0]
test_text = [text[j] for i, j in enumerate(random_order) if i % 10 == 0]
train_stance = [stance[j] for i, j in enumerate(random_order) if i % 10 != 0]
test_stance = [stance[j] for i, j in enumerate(random_order) if i % 10 == 0]

np.savetxt(os.path.join(save_file_path,'train_topic.txt'), train_topic, fmt = "%s", delimiter = " ")
np.savetxt(os.path.join(save_file_path,'test_topic.txt'), test_topic, fmt = "%s", delimiter = " ")
np.savetxt(os.path.join(save_file_path,'train_text.txt'), train_text, fmt = "%s", delimiter = " ")
np.savetxt(os.path.join(save_file_path,'test_text.txt'), test_text, fmt = "%s", delimiter = " ")
np.savetxt(os.path.join(save_file_path,'train_stance.txt'), train_stance, fmt = "%s", delimiter = " ")
np.savetxt(os.path.join(save_file_path,'test_stance.txt'), test_stance, fmt = "%s", delimiter = " ")