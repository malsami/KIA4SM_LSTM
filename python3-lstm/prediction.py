import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import csv

# PKG has a fixed set of labels. Integer encoding is used where integer # value is assigned to each label
PKGs = {
    'pi': 0,
    'hey': 1,
    'tumatmul': 2,
    'cond_mod': 3
}

# Integer encoding for Exit_Values from Jobs
Exit_Values = {
    'EXIT': 1,
    'EXIT_CRITICAL': 0,
    'EXIT_PERIOD': 2,
    'OUT_OF_CAPS': 3,
    'OUT_OF_QUOTA': 4,
    'EXIT_ERROR': 5
}

# ARG values ranged from 1 to 205.891.132.094.649, these values were normalized and scaled # to range from 1 to 17
Arg_Values = {
    1: 1,
    4096: 2,
    8192: 3,
    16384: 4,
    32768: 5,
    65536: 6,
    131072: 7,
    262144: 8,
    524288: 9,
    1048576: 10,
    2097152: 11,
    847288609443: 12,
    2541865828329: 13,
    7625597484987: 14,
    22876792454961: 15,
    68630377364883: 16,
    205891132094649: 17
}

print("Doing writing")

DB_PATH = "/home/bernhard/panda_v4.db"
TASKS_DICT = {}


def taskToFeatureList(task):
    # returns a fature list for the corresponding task values
    feature = []
    feature.append(task['Priority'])
    feature.append(task['Period'])
    feature.append(task['Number_of_Jobs'])
    feature.append(task['PKG'])
    feature.append(task['Arg'])
    feature.append(task['CRITICALTIME'])
    return feature


def getTaskFeatures(db_path):  # c is the cursor for the db
    # returns a dictionary
    # { task_id : [ feature, list ]
    conn = sqlite3.connect(db_path)
    conn.row_factory = lambda C, R: {c[0]: R[i] for i, c in enumerate(C.description)}
    db_cursor = conn.cursor()
    db_cursor.execute('select Task_ID,Priority,Period,PKG,Arg,CRITICALTIME,Number_of_Jobs from Task')
    outputTable = db_cursor.fetchall()

    tasks_dict = {}
    for row in outputTable:
        row['Period'] = int(row['Period'] / 1000)
        row['Number_of_Jobs'] = int(row['Number_of_Jobs'])
        row['PKG'] = PKGs[row['PKG']]
        row['CRITICALTIME'] = int(row['CRITICALTIME'] / 1000)
        row['Arg'] = Arg_Values[row['Arg']]
        tasks_dict[row['Task_ID']] = taskToFeatureList(row)
    return tasks_dict


def processTaskset(tasksetData):
    # tasksetData is a list of tuples returned from the DB in getTasksetData()
    label = tasksetData[0][-1]
    features = []
    jobExitsByTask = {}
    for tsData in tasksetData:
        try:
            jobExitsByTask[tsData[4]].append(Exit_Values[tsData[5]])
        except KeyError:
            jobExitsByTask[tsData[4]] = [Exit_Values[tsData[5]]]
    for taskIdNo in (1, 2, 3):
        if tasksetData[0][taskIdNo] != -1:
            features += TASKS_DICT[tasksetData[0][taskIdNo]]
            try:
                features += jobExitsByTask[tasksetData[0][taskIdNo]]
            except KeyError:
                features += [Exit_Values['EXIT_ERROR']]
    return np.array(features), label


def getFeaturesLabels(db_path):
    conn = sqlite3.connect(db_path)
    db_cursor = conn.cursor()
    command = 'SELECT TaskSet.Set_ID, TaskSet.TASK1_ID, TaskSet.TASK2_ID, TaskSet.TASK3_ID, Job.Task_ID, Job.Exit_Value, TaskSet.Successful' \
              ' FROM TaskSet JOIN Job' \
              ' ON TaskSet.Set_ID = Job.Set_ID and' \
              ' (TaskSet.TASK1_ID == Job.Task_ID or' \
              ' TaskSet.TASK2_ID == Job.Task_ID or' \
              ' TaskSet.TASK3_ID == Job.Task_ID);'
    db_cursor.execute(command)
    # data_table format: [( TaskSet.Set_ID, TaskSet.TASK1_ID, TaskSet.TASK2_ID, TaskSet.TASK3_ID, Job.Task_ID, Job.Exit_Value, TaskSet.Successful)]
    data_table = db_cursor.fetchall()
    print('reading taskset_jobs join done')
    print('The current time is:', datetime.now())
    finalFeatureList = []
    finalLabelList = []
    currentTset = data_table[0][0]  # first taskset id
    tSetJobs = []
    totalSize = len(data_table)
    for row in data_table:
        if row[0] % 1000 == 0:
            print('processed', int(100 * (row[0] / totalSize)), '%')
        if row[0] == currentTset:
            # then still same setTset
            tSetJobs.append(row)
        else:
            # job of next taskset
            # process data and record new
            features, label = processTaskset(tSetJobs)
            finalFeatureList.append(features)
            finalLabelList.append(label)
            tSetJobs = []
            currentTset = row[0]
            tSetJobs.append(row)
    # proess last taskset
    features, label = processTaskset(tSetJobs)
    finalFeatureList.append(features)
    finalLabelList.append(label)
    return finalFeatureList, finalLabelList


TASKS_DICT = getTaskFeatures(DB_PATH)
print('Tasks have been added to TASKS_DICT')
print('length of taskdict: ', len(TASKS_DICT))
print('example task 222:', TASKS_DICT[222])

features, labels = getFeaturesLabels(DB_PATH)

print('The current time is:', datetime.now())

print("Done reading")

labels = np.array(labels)  # to save the labels list as numpy array

# To make a fixed length vector, if the vector is smaller than 56 then replace the empty values with -1. if longer than 56 trim the value
features = pad_sequences(features, maxlen=56, value=-1, padding='post', truncating='post')

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
