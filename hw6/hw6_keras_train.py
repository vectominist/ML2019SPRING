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

from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(7)

# import emoji
import jieba
from gensim.models import Word2Vec

# import matplotlib.pyplot as plt # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_data_x(file_x, dict_path):
    # Read in all training data
    print('Reading data X...')
    x_train = pd.read_csv(file_x, delimiter=',', encoding='UTF8').to_numpy()
    x_train = x_train[:, 1:]
    # x_train shape : (120000, 1)

    # print('Demojizing...')
    removeP = string.punctuation + '？！、，。：；—「」『』～⋯…”“‘（）♥️⁽⁎❝ω❝ົཽ⁎ω•́๑∀'
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

    # Character Level
    # seg_list = [list(x_train[i][0]) for i in range(len(x_train))]
    # return seg_list

    # Word Segmentation by Jeiba 
    jieba.load_userdict(dict_path)
    print('Segmenting...')
    seg_list = [jieba.lcut(x_train[i][0]) for i in range(len(x_train))]
    
    print('Finished reading data X...')
    return seg_list

def read_data_y(file_y):
    # Read in all training data
    print('Reading data Y...')
    y_train = pd.read_csv(file_y, delimiter=',', encoding='UTF8').to_numpy().astype(int)
    y_train = np.array(y_train[:, 1], dtype=float) # int or float
    # y_train shape : (120000,)

    print('Finished reading data Y...')
    return y_train


def train_Word2Vec_model(x_sentences, minCount=10, Size=100, iters=20, wndw=3):
    print('Training Word2Vec model...')
    W_model = Word2Vec(min_count=minCount, 
                        size=Size, 
                        window=wndw, 
                        workers=4,
                        sg=1,
                        batch_words = 64)
    t = time.time()
    W_model.build_vocab(x_sentences, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))
    W_model.train(x_sentences, total_examples=W_model.corpus_count, epochs=iters, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))

    return W_model

def try_WModel(model, query):
    try:
        print("\n---- %s 相似詞前 20 排序 ----" % (query))
        res = model.most_similar(query, topn = 20)
        for item in res:
            print(item[0]+","+str(item[1]))
    except Exception as e:
        print(repr(e))


def val_split(data, labels, split_ratio):
    randomize = np.arange(len(data))
    randomize = np.random.shuffle(randomize)
    v_size = int(math.floor(len(data) * split_ratio))
    X_train, Y_train = data[0:v_size], labels[0:v_size]
    X_valid, Y_valid = data[v_size:], labels[v_size:]
    return X_train, Y_train, X_valid, Y_valid


def build_model_3(in_L, in_D, word_model):
    embedding_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size))
    word_list = {}
    vocab_list = [(word, word_model.wv[word]) for word, vector in word_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vec = vocab
        embedding_matrix[i + 1] = vec
        word_list[word] = i + 1
    
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], 
                        output_dim=embedding_matrix.shape[1], 
                        weights=[embedding_matrix], 
                        trainable=False))
    model.add(LSTM(128,activation="tanh",dropout=0.3,return_sequences = True,
            kernel_initializer='Orthogonal'))
    model.add(Bidirectional(LSTM(128,activation="tanh",dropout=0.3,return_sequences = False,
            kernel_initializer='Orthogonal')))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    check_point = ModelCheckpoint('best_3.h5', monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='max')
    return model, check_point, early_stop, word_list

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

# Parameters
EMB_SIZE = 128
mn_cnt = 3
w_iters = 27
w_wndw = 5
w_smpl = 0.001
w_ap = 0.025
w_m_ap = 0.0001
w_ngt = 5


batch_size = 1024
epochs = 100
ulen = 80
side = 0
split_r = 0.85

if __name__ == "__main__":
    # Easy run : python hw6_keras_train.py data/train_x.csv data/train_y.csv data/test_x.csv data/dict.txt.big
    # argv 1 : data/train_x.csv
    # argv 2 : data/train_y.csv
    # argv 3 : data/test_x.csv
    # argv 4 : data/dict.txt.big

    # word_model = Word2Vec.load('models/wmodel')
    # try_WModel(word_model, '八九')
    
    x_train = read_data_x(sys.argv[1], sys.argv[4])
    y_train = read_data_y(sys.argv[2])
    tlen = len(x_train)

    x_train = x_train[:119018]
    y_train = y_train[:119018]

    x_test = read_data_x(sys.argv[3], sys.argv[4])
    word_model = train_Word2Vec_model(x_train + x_test, 
                                        minCount=mn_cnt, 
                                        Size=EMB_SIZE, 
                                        iters=w_iters, 
                                        wndw=w_wndw)
    word_model.save('wmodel_rnn')
    # while(True):
    #     qy = sys.stdin.readline()
    #     qy = qy.split()
    #     try_WModel(word_model, qy[0])
    #     if qy[0] == '0':
    #         break
    del x_test

    word_model = Word2Vec.load('wmodel_rnn')
    # x_train = get_Word2Vec(x_train, word_model, ulen=ulen, emb_size=EMB_SIZE, side=side)

    
    model, check_point, early_stop, word_list = build_model_3(ulen, EMB_SIZE, word_model)
    x_train = text2index(x_train, word_list, ulen)
    x_train, y_train, x_val, y_val = val_split(x_train, y_train, split_r)

    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, \
    callbacks = [check_point, early_stop], validation_data = (x_val, y_val))
    model.save('model.h5')
    del model

    # # list all data in history
    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
