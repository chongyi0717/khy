import pandas as pd
TRAIN_FILE_PATH = 'input/train.csv'
TEST_FILE_PATH = 'input/test.csv'
pd1 = pd.read_csv(TRAIN_FILE_PATH)
pd2 = pd.read_csv(TEST_FILE_PATH)
X_train  =  pd1['Title']+' '+pd1['Description'] # also removing the class from the training dataset

X_test   =  pd2['Title']+'  '+pd2['Description'] # also removing the class from the training dataset

#把class index作為結果
y_train  =   pd1['Class Index'].apply(lambda x: x-1)  # assigning label of train

y_test =    pd2['Class Index'].apply(lambda x: x-1) # assigning lale of test

#看訓練集裡面最大的字數是多少
max_len = X_train.map(lambda x : len(x.split())).max()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocabulary_size = 10000 # random value
embed_size      = 32    # random value 

tok = Tokenizer(num_words=vocabulary_size) #可以把每個字符算出現的次數，num_words是最大出現次數
tok.fit_on_texts(X_train.values) #每個詞都可以變成一個對應的數字

#每一篇文章就可以用一維陣列來表示
X_train = tok.texts_to_sequences(X_train) 
X_test  = tok.texts_to_sequences(X_test)


# # Now we need to pad all the sequences based on the max value 
#把陣列都變成相同維度，預設都是用0來替補，max_len可以用來決定每個維度的長度
X_train = pad_sequences(X_train,maxlen=max_len)
X_test = pad_sequences(X_test,maxlen=max_len)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
import pandas as pd
import numpy as np

#Data Visualization
import matplotlib.pyplot as plt

#Text Color
from termcolor import colored

#Train Test Split
from sklearn.model_selection import train_test_split

#Model Evaluation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from mlxtend.plotting import plot_confusion_matrix

#Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Embedding
vocabulary_size = 10000 # random value
embed_size      = 32  # random value
model = Sequential()
model.add(Embedding(vocabulary_size,embed_size,input_length = max_len)) #input layer is embedding layer
model.add(Bidirectional(LSTM(128, return_sequences=True)))              # Bidirectinal LSTM
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(GlobalMaxPooling1D())                                         # Flattening layer to reduce everything in a vector form
model.add(Dense(256, activation='relu'))                                                  # Dense layer
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.25))                                                # doing regularization in Neural Network
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(4, activation='softmax'))                               #  we have 4 labels as output

model_name='model/model.h5'
if os.path.exists(model_name):
    model.load_weights(model_name)
else:
    model.compile(loss = 'sparse_categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy']             
                )
    model.fit(X_train,y_train,batch_size=256,validation_data=(X_test,y_test),epochs=20)
    model.save(model_name)


def modelDemo(news_text):
        
    #News Labels
    labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']

    test_seq = pad_sequences(tok.texts_to_sequences(news_text), maxlen=max_len)

    test_preds = [labels[np.argmax(i)] for i in model.predict(test_seq)]

    for news, label in zip(news_text, test_preds):
        # print('{} - {}'.format(news, label))
        print('{}-{}'.format(colored(label, 'blue')))
        
preds = [np.argmax(i) for i in model.predict(X_test)]
print("Recall of the model is {:.2f}".format(recall_score(y_test, preds, average='micro')))
print("Precision of the model is {:.2f}".format(precision_score(y_test, preds, average='micro')))
print("Accuracy of the model is {:.2f}".format(accuracy_score(y_test, preds)))