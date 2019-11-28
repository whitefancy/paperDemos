X1, X2, Y = [], [], []
for s in train_sentences:
    indices, segments = tokenizer.encode(first=s['text'][0][:100], second=s['aspectTerm'][0], max_len=maxlen)
    X1.append(indices)
    X2.append(segments)
    if s['polarity'][0] == 'neutral':
        y_val = 0
    elif s['polarity'][0] == 'negative':
        y_val = 1
    elif s['polarity'][0] == 'positive':
        y_val = 2
    elif s['polarity'][0] == 'conflict':
        y_val = 3
    Y.append([y_val])
X1 = seq_padding(X1)
X2 = seq_padding(X2)
Y = seq_padding(Y)
[X1, X2], Y