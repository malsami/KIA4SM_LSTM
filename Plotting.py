from keras.models import load_model
from keras.utils import plot_model

model = load_model ( 'My_LSTM_Model.h5' )
# Plotting the model with Graphgiz
plot_model ( model, to_file='model.png', show_shapes=True, show_layer_names=True )
