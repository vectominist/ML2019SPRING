import sys
import csv
import time
import string
import re
import math
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, LSTM, CuDNNLSTM, Bidirectional, TimeDistributed
from keras.layers import Dropout, BatchNormalization, Input, Flatten, Embedding
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import regularizers, constraints, initializers
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import jieba
from gensim.models import Word2Vec

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_data_x(file_x, dict_path):
    # Read in all training data
    print('Reading data X...')
    x_train = pd.read_csv(file_x, delimiter=',', encoding='UTF8').to_numpy()
    x_train = x_train[:, 1:]
    # x_train shape : (120000, 1)

    # print('Demojizing...')
    RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    RE_SAVE = re.compile('[^\u4e00-\u9fa5^\u3105-\u3129^ ^,^.^!^?^，^。^？^！^a-z^A-Z^0-9]', flags=re.UNICODE)
    translator_1   = str.maketrans(removeP, ' ' * len(removeP))
    translator_2   = str.maketrans(':', ' ')
    translator_3   = str.maketrans('ㄅㄆㄇㄉㄊㄋㄌㄏㄐㄛㄩㄧ.', '吧炮媽的他呢了呵雞喔約一點')
    translator_3_1 = str.maketrans('ㄆㄊㄐㄩ+', '炮他雞約加')
    translator_4   = str.maketrans('吧呢啊喔哦嗯啦嗎阿噢呦喲唷的了很超蠻欸唉', '                    ')
    translator_5   = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ，。？！', 'abcdefghijklmnopqrstuvwxyz,.?!')
    translator_6   = str.maketrans('123456789', '一二三四五六七八九')
    
    for i in range(len(x_train)):
        progress = ('#' * int(float(i)/len(x_train)*40)).ljust(40)
        print ('Preprocessing [%06d/%06d] | %s |' % (i+1, len(x_train), \
                progress), end='\r', flush=True)
        # print(x_train[i, 0])
        # x_train[i, 0] = RE_EMOJI.sub(r'', x_train[i, 0])
        # x_train[i, 0] = x_train[i, 0].translate(translator_3)
        x_train[i, 0] = RE_SAVE.sub('', x_train[i, 0])
        # x_train[i, 0] = x_train[i, 0].translate(translator_4)
        x_train[i, 0] = x_train[i, 0].translate(translator_5)
        # x_train[i, 0] = x_train[i, 0].translate(translator_1)
        # x_train[i, 0] = ' '.join(x_train[i, 0].split())
        # x_train[i, 0] = emoji.demojize(x_train[i, 0])
        # x_train[i, 0] = x_train[i, 0].translate(translator_2)
        # x_train[i, 0] = ' '.join(x_train[i, 0].split())
        
        # if (x_train[i, 0][0] == 'B' or x_train[i, 0][0] == 'b') and len(x_train[i, 0]) > 1:
        #     cnt = 1
        #     while cnt < len(x_train) and ord(x_train[i, 0][cnt]) >= ord('0') and ord(x_train[i, 0][cnt]) <= ord('9'):
        #         cnt += 1
        #     x_train[i, 0] = x_train[i, 0][cnt:]
        for j in range(500, -1, -1):
            x_train[i, 0] = x_train[i, 0].replace('b' + str(j), '')
        # x_train[i, 0] = x_train[i, 0].replace('0000', '萬').replace('000', '千').replace('00', '百').replace('0', '十')
        # x_train[i, 0] = x_train[i, 0].translate(translator_6)
        # x_train[i, 0] = x_train[i, 0].replace(' ', '')
        
        # print(x_train[i, 0])
        # input()
    progress = ('#' * 40).ljust(40)
    print ('Preprocessing [%06d/%06d] | %s |  ' % (len(x_train), len(x_train), progress), flush=True)


    # Word Segmentation by Jeiba 
    jieba.load_userdict(dict_path)
    print('Segmenting...')
    seg_list = [jieba.lcut(x_train[i][0]) for i in range(len(x_train))]
    
    print('Finished reading data X...')
    return seg_list



def text2index(data, w_list, ulen):
    index_data = []
    for s in data:
        new_s = []
        for word in s:
            try:
                new_s.append(w_list[word])
            except:
                new_s.append(0)
        index_data.append(new_s)
    index_data = pad_sequences(index_data, maxlen=ulen, padding='pre')
    return np.array(index_data)

def test_RNN(model, x_test, file_path):
    # Calculate predicted answer
    ans = []
    
    y_ans = model.predict(x_test)
    for i in range(len(x_test)):
        ans.append([str(i)])
        a = y_ans[i][0]
        if a > 0.5 - 1e-10:
            a = int(1)
        else:
            a = int(0)
        ans[i].append(a)
        

    text = open(file_path, 'w+')
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(['id','label'])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()

def ensemble_RNN(models, x_test, file_path):
    # Calculate predicted answer
    ans = []
    m_len = len(models)
    y_ans = []
    for i in range(m_len):
        y_ans.append(models[i].predict(x_test))
    
    for i in range(len(x_test)):
        ans.append([str(i)])
        a = 0
        for j in range(m_len):
            a += y_ans[j][i][0]
        a /= m_len
        # print('-- %d : %.6f --' % (i, a))
        # input()
        if a > 0.5 - 1e-10:
            a = int(1)
        else:
            a = int(0)
        ans[i].append(a)
        

    text = open(file_path, 'w+')
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(['id','label'])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()

# Parameters
EMB_SIZE = 128
ulen = 80

if __name__ == "__main__":
    # Easy run  : python hw6_keras_test.py data/test_x.csv data/dict.txt.big result/prediction.csv
    # Easy test : python hw6_keras_test.py data/my_test.csv data/dict.txt.big result/prediction.csv
    # argv 1 : data/test_x.csv
    # argv 2 : data/dict.txt.big
    # argv 3 : result/prediction.csv

    x_test = read_data_x(sys.argv[1], sys.argv[2])
    tlen = len(x_test)

    word_model = Word2Vec.load('wmodel_rnn')
    embedding_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size))
    word_list = {}
    vocab_list = [(word, word_model.wv[word]) for word, vector in word_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vec = vocab
        embedding_matrix[i + 1] = vec
        word_list[word] = i + 1
    x_test = text2index(x_test, word_list, ulen)
    
    # Read single model
    # model = load_model('models/best_3_75444.h5')
    # test_RNN(model, x_test, sys.argv[3])

    # Ensemble models
    models = []
    models.append(load_model('best_3_75444.h5'))
    models.append(load_model('best_3_75332.h5'))
    models.append(load_model('best_3_75433.h5'))
    models.append(load_model('best_3_75181.h5'))
    models.append(load_model('best_3G_75181.h5'))
    ensemble_RNN(models, x_test, sys.argv[3])
