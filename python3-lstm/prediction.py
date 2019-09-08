import numpy as np
import pickle
from keras.models import load_model
import csv

with open ( '56_features', 'wb' ) as outfile:  # 'wb' is the file mode, it means 'write binary'
    #pickle.load(outfile features, outfile)
    features = pickle.load(outfile, *, fix_imports=True, encoding="ASCII", errors="strict")

with open ( '56_labels', 'wb' ) as outfile:
    labels = pickle.load(outfile, *, fix_imports=True, encoding="ASCII", errors="strict")
    #pickle.dump(labels, outfile)

model = load_model('My_LSTM_Model.h5')
X = np.expand_dims(features, axis=2)
preds = model.predict(X)

arr = []
for i in range(len(labels)):
    l = labels[i]
    p = np.argmax(preds[i])
    print ( "the actual value is{0}and the predicted value is {1}".format(l, p))
    arr.append([i + 1, l, p])

csvfile = "Predicion_results.csv"

i = 0
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    if (i == 0):
        writer.writerow(["TaskSet ID", "Actual Value", "Predicted Value"])
    i += 1
    writer.writerows(arr)
