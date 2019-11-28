#按照不同的主题分割数据
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

top =list(set(topic))
for ix,x in enumerate(top):
    text_i=[text[it] for it, t in enumerate(topic) if t == top[ix]]
    topic_i=[topic[it] for it, t in enumerate(topic) if t == top[ix]]
    stance_i=[stance[it] for it, t in enumerate(topic) if t == top[ix]]
    random_order = np.linspace(0,len(stance_i)-1,num=len(stance_i)-1,dtype=int)
    np.random.shuffle(random_order)
    train_topic = [topic_i[j] for i, j in enumerate(random_order) if i % 10 != 0]
    test_topic = [topic_i[j] for i, j in enumerate(random_order) if i % 10 == 0]
    train_text = [text_i[j] for i, j in enumerate(random_order) if i % 10 != 0]
    test_text = [text_i[j] for i, j in enumerate(random_order) if i % 10 == 0]
    train_stance = [stance_i[j] for i, j in enumerate(random_order) if i % 10 != 0]
    test_stance = [stance_i[j] for i, j in enumerate(random_order) if i % 10 == 0]
    np.savetxt(os.path.join(save_file_path,'train_topic'+str(ix)+'.txt'), train_topic, fmt = "%s", delimiter = " ")
    np.savetxt(os.path.join(save_file_path,'test_topic'+str(ix)+'.txt'), test_topic, fmt = "%s", delimiter = " ")
    np.savetxt(os.path.join(save_file_path,'train_text'+str(ix)+'.txt'), train_text, fmt = "%s", delimiter = " ")
    np.savetxt(os.path.join(save_file_path,'test_text'+str(ix)+'.txt'), test_text, fmt = "%s", delimiter = " ")
    np.savetxt(os.path.join(save_file_path,'train_stance'+str(ix)+'.txt'), train_stance, fmt = "%s", delimiter = " ")
    np.savetxt(os.path.join(save_file_path,'test_stance'+str(ix)+'.txt'), test_stance, fmt = "%s", delimiter = " ")