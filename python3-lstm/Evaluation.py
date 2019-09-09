import matplotlib.pyplot as plt
import seaborn as sns;

sns.set ()
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
import pickle
import numpy as np
from keras.models import load_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

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
        newy.append ( np.array ( [0, 1] ) )
    else:
        newy.append ( np.array ( [1, 0] ) )
y = np.array ( newy )
print ( count )

X_train, X_test, y_train, y_test = train_test_split ( X, y, test_size=0.3, random_state=42)
print ( X_train.shape )

model = load_model ( 'My_LSTM_Model.h5' )  # loading saved model

eva = loss, accuracy = model.evaluate ( X, y )

y_preds = model.predict ( X_test )

yp = []
yt = []
i = 0
for p in y_preds:
    yp.append ( np.argmax ( p ) )
    yt.append ( np.argmax ( y_test[i] ) )
    i += 1

print ( "F1 score is: ", f1_score ( yt, yp, average='micro' ) )
print ( classification_report ( yt, yp ) )
print ( accuracy_score ( yt, yp ) )

cm = metrics.confusion_matrix ( yt, yp )
cm_df = pd.DataFrame ( cm,
                       index=['1', '0'],
                       columns=['1', '0'] )

plt.figure ( figsize=(5.5, 4) )
sns.heatmap ( cm_df, annot=True, fmt='g' )
plt.title ( 'Confusion Matrix \n Accuracy:{0:.3f}'.format ( accuracy_score ( yt, yp ) ) )
plt.ylabel ( 'True label' )
plt.xlabel ( 'Predicted label' )
plt.show ()
plt.savefig ( 'Confusion_Matrix.png' )
