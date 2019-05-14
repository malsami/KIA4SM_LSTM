import pickle
import time
import tensorboard
from keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
import sys

np.set_printoptions ( threshold=sys.maxsize )

# from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

name = "42_-{}".format ( int ( time.time () ) )

tensorboard = TensorBoard ( log_dir="logs42/{}".format ( name ), write_graph=True, write_grads=False,
                            write_images=False, update_freq='epoch' )
es = EarlyStopping ( monitor='val_loss', mode='min', verbose=1 )

with open ( '42_features', 'rb' ) as fp:
    X = pickle.load ( fp )

with open ( '42_labels', 'rb' ) as fp:
    y = pickle.load ( fp )

X = np.expand_dims ( X, axis=2 )
newy = []
count = 0

for val in y:
    if val == 0:
        count += 1
        newy.append ( np.array ( [1, 0] ) )
    else:
        newy.append ( np.array ( [0, 1] ) )
y = np.array ( newy )
print ( count )

X_train, X_test, y_train, y_test = train_test_split ( X, y, test_size=0.3 )
input = Input ( shape=(42, 1) )

fi = 0
arr = []  ## to save the trials
import csv

csvfile = "optimization_oneLSTM_oneDense.csv"

with open ( csvfile, "w" ) as csvoutput:
    writer = csv.writer ( csvoutput, lineterminator='\n' )
    writer.writerow ( ["Trial number", "LSTM size", "Dense size", "Dropout Ratio", "Batch Size", "Activation Function",
                       "Number of Epoch", "Accuracy", "Best Val_Loss"] )
    # fi+= 1

dense_layer_sizes = [64, 128, 256, 512]
layer_sizes = [64, 128, 256]
n_dropout = [0, 0.2, 0.4, 0.6]
batch_s = [16, 32, 64, 128]

input = Input ( shape=(42, 1) )
print ( input.shape )

for i in range ( 0, len ( layer_sizes ) ):

    for s in range ( 0, len ( dense_layer_sizes ) ):

        for dro in range ( 0, len ( n_dropout ) ):

            for bs in range ( len ( batch_s ) ):

                NAME = "{}-1st-LSTM-{}-1st-dense-dropout-{}-batch-{}-FI{}".format ( layer_sizes[i],
                                                                                    dense_layer_sizes[s],
                                                                                    n_dropout[dro], batch_s[bs], fi )
                tensorboard = TensorBoard ( log_dir="logs/{}".format ( NAME ) )

                lstm_model = CuDNNLSTM ( layer_sizes[i], return_sequences=True ) ( input )
                lstm_model = CuDNNLSTM ( layer_sizes[i] * 2 ) ( input )

                if (layer_sizes[i]) == s:

                    d = Dense ( dense_layer_sizes[s], activation='relu' ) ( lstm_model )
                    d = Dropout ( n_dropout[dro] ) ( d )

                    d = Dense ( dense_layer_sizes[s] * 2, activation='relu' ) ( d )
                    d = Dropout ( n_dropout[dro] ) ( d )

                    d = Dense ( dense_layer_sizes[s] * 4, activation='relu' ) ( d )
                    d = Dropout ( n_dropout[dro] ) ( d )

                    d = Dense ( dense_layer_sizes[s] * 8, activation='relu' ) ( d )
                    d = Dropout ( n_dropout[dro] ) ( d )

                else:

                    d = Dense ( dense_layer_sizes[i] * 2, activation='relu' ) ( lstm_model )
                    d = Dropout ( n_dropout[dro] ) ( d )

                    d = Dense ( dense_layer_sizes[i] * 4, activation='relu' ) ( d )
                    d = Dropout ( n_dropout[dro] ) ( d )

                    d = Dense ( dense_layer_sizes[i] * 8, activation='relu' ) ( d )
                    d = Dropout ( n_dropout[dro] ) ( d )

                    d = Dense ( dense_layer_sizes[i] * 16, activation='relu' ) ( d )
                    d = Dropout ( n_dropout[dro] ) ( d )

                output = Dense ( 2, activation='softmax' ) ( d )

                model = Model ( inputs=input, outputs=output )

                adam = Adam ( lr=0.0001 )

                model.compile ( optimizer=adam, loss='binary_crossentropy', metrics=['acc'] )

                model.fit ( X_train, y_train, batch_size=batch_s[bs], epochs=100, validation_data=(X_test, y_test),
                            callbacks=[tensorboard, es] )

                model.save ( '{}.h5'.format ( NAME ) )

                when = es.stopped_epoch
                bes = es.best
                fi += 1
                arr.append (
                    [fi, layer_sizes[i], dense_layer_sizes[s], n_dropout[dro], batch_s[bs], 'relu', when, bes] )

                with open ( csvfile, "a",
                            newline='' ) as csvoutput:  # 'a' parameter allows you to append to the end of the file instead of simply overwriting the existing content
                    # with open(csvfile, "w") as csvoutput:
                    writer = csv.writer ( csvoutput )
                    writer.writerows ( arr )





