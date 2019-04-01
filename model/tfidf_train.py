import numpy as np
import pandas as pd
import pickle

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, SimpleRNN
from keras.models import load_model

TRAIN_PATH = 'train2.pkl'
X_TRAIN = "x_train.pkl"
Y_TRAIN = "y_train.pkl"
VEC_SIZE = 1000
VEC_DIM = 5

dtrain = pd.read_pickle(TRAIN_PATH)
tfidf = []
thread_ids = dtrain.thread_no.unique()

for id in thread_ids :
    d = dtrain[dtrain['thread_no'] == id]
    tfidf.append(d['bodyidx'])

print(len(tfidf))


def fill_vec_train(thread_embed, start, end) :
    x_train = [[0 for x in range (VEC_SIZE)] for y in range(VEC_DIM)]
    idx = 0
    for x in range(start, end) :
        x_train[idx] = thread_embed[x]
        idx += 1
    return x_train


def max(i, j) :
    if i >= j :
        return i
    else :
        return j


x_train = []
y_train = []

def gen_x_y() :
    for thread_embed in tfidf :
        temp = []
        for x in thread_embed :
            temp.append(np.array(x).tolist())

        for x in range (len(temp) - 1) :
            start = max(0, x - 5)
            end = x
            x_train.append(fill_vec_train(temp, start, end))
            y_train.append(temp[x + 1])

gen_x_y()

# print(len(x_train))
# print(len(x_train[0][0]))

pickle.dump(x_train, open(X_TRAIN, "wb"))
pickle.dump(y_train, open(Y_TRAIN, "wb"))


# x_train = pickle.load(open(X_TRAIN, "rb"))
# y_train = pickle.load(open(Y_TRAIN, "rb"))

# print(len(x_train))
# print(x_train[0])

SPLIT_DATA = 36
x_test = np.array(x_train[-SPLIT_DATA:])
# x_test = x_test[:,:,np.newaxis]
y_test = np.array(y_train[-SPLIT_DATA:])
x_train = np.array(x_train[:-SPLIT_DATA])
# x_train = x_train[:,:,np.newaxis]
y_train = np.array(y_train[:-SPLIT_DATA])
print(x_test.shape)

print("length of x_train : " + str(len(x_train)))
print("length of y_train : " + str(len(y_train)))
print(x_train.shape)
print(y_train.shape)

model = Sequential()
# model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(SimpleRNN(1000, input_shape=(VEC_DIM, VEC_SIZE)))
# model.add(Dropout(0.5))
model.add(Dense(1000, activation='sigmoid'))

model.compile('adagrad', loss='mean_squared_error', metrics=['accuracy'])


# inp = Input(shape=(VEC_SIZE,))
# # x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
# x = LSTM(1000)(inp)
# # x = GlobalMaxPool1D()(x)
# x = Dense(1000, activation="relu")(x)
# x = Dropout(0.25)(x)
# x = Dense(1000, activation="sigmoid")(x)

# model = Model(inputs=inp, outputs=x)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
print("model fitting!!")
model.fit(x_train, y_train, validation_data=[x_test, y_test])
print("predicting!")
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# model.predict(x_train[0])