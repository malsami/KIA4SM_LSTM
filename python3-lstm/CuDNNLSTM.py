import pickle
import time
import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

name = "logname-{}".format ( int ( time.time () ) )

# both metrics and early stopping conditions are defined here and then saved in the log42 file
tensorboard = TensorBoard ( log_dir="logs42/{}".format ( name ), write_graph=True, write_grads=False,
                            write_images=False, update_freq='epoch' )
es = EarlyStopping ( monitor='val_loss', mode='min', verbose=1 )  # define early stopping criteria

# Importing the the extracted features and labels
with open ( '56_features', 'rb' ) as fp:
    X = pickle.load ( fp )
with open ( '56_labels', 'rb' ) as fp:
    y = pickle.load ( fp )

# LSTM’s input shape argument expects a three-dimensional array as an input in this order: Samples, timestamps and features. This is why we need to add another dimention to the numpy array.
X = np.expand_dims ( X, axis=2 )
newy = []
count = 0

# The output from label should be [1,0] or [0,1] while SoftMax is implemented.
for val in y:
    if val == 0:
        count += 1
        newy.append ( np.array ( [0, 1] ) )
    else:
        newy.append ( np.array ( [1, 0] ) )
y = np.array ( newy )
# print ( count )

# devide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split ( X, y, test_size=0.3 ,random_state=42)
# print ( X_train.shape )

# LSTM input is fifty-six time-steps and one feature at each time-step is represented by the notation: (56,1).
input = Input ( shape=(56, 1) )

# the first LSTM layer has 64 cells, the number must be equal/bigger than the input size. If you are using a CPU then change CuDNNLSTM to LSTM
lstm = CuDNNLSTM ( 64, return_sequences=True ) (
    input )  # Return_sequences is set true because the first LSTM has to return a sequence, which then can be fed into the 2nd LSTM
lstm = CuDNNLSTM ( 128, return_sequences=True ) ( lstm )
lstm = CuDNNLSTM ( 256 ) ( lstm )

# the first dense-lyer has 128 neurons its activation function is relu which chooses the max(0,previous layer’s value)
d = Dense ( 256, activation='relu' ) ( lstm )
d = Dropout ( 0.3 ) ( d )
d = Dense ( 512, activation='relu' ) ( d )
d = Dropout ( 0.3 ) ( d )
d = Dense ( 1024, activation='relu' ) ( d )
d = Dropout ( 0.3 ) ( d )

# Network compilation


# 2 output units
output = Dense ( 2, activation='softmax' ) ( lstm )
model = Model ( inputs=input, outputs=output )

# loss function used is binary cross entropy,it is used for binary classification problem
adam = Adam ( lr=0.00001 )
model.compile ( optimizer=adam, loss='binary_crossentropy', metrics=['acc'] )

# data is sent to the model in the batches of 64, 200 iterations are done.
model.fit ( X_train, y_train, batch_size=64, epochs=200, validation_data=(X_test, y_test), callbacks=[tensorboard, es] )

model.save ( 'My_LSTM_Model.h5' )  # Save the model
