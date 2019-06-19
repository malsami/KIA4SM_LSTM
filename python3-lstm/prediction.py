import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import csv

df_taskset = pd.read_csv ( 'TaskSet.csv' )
# df_taskset = df_taskset.sample(frac=0.0001, random_state=99)
df_task = pd.read_csv ( 'Task.csv' )
df_job = pd.read_csv ( 'Job.csv' )

ntn = df_task[['PKG']].values
ntn1 = []
for n in ntn:
    ntn1.append ( n[0] )

PKGs = {}
PKGs['pi'] = 0
PKGs['hey'] = 1
PKGs['tumatmul'] = 2
PKGs['cond_mod'] = 3

Exit_Values = {}
Exit_Values['EXIT'] = 1
Exit_Values['EXIT_CRITICAL'] = 0

Arg_Values = {}
Arg_Values[1] = 1
Arg_Values[4096] = 2
Arg_Values[8192] = 3
Arg_Values[16384] = 4
Arg_Values[32768] = 5
Arg_Values[65536] = 6
Arg_Values[131072] = 7
Arg_Values[262144] = 8
Arg_Values[524288] = 9
Arg_Values[1048576] = 10
Arg_Values[2097152] = 11
Arg_Values[847288609443] = 12
Arg_Values[2541865828329] = 13
Arg_Values[7625597484987] = 14
Arg_Values[22876792454961] = 15
Arg_Values[68630377364883] = 16
Arg_Values[205891132094649] = 17

i = 0
features = []
labels = []
with tqdm(total=len(list(df_taskset.iterrows()))) as pbar:
    for index, row in df_taskset.iterrows():

        try:

            i += 1
            grid = int(df_taskset.loc[index, 'Set_ID'])
            res = int(df_taskset.loc[index, 'Successful'])
            print(grid)
            first_task = int(df_taskset.loc[index, 'TASK1_ID'])
            second_task = int(df_taskset.loc[index, 'TASK2_ID'])
            third_task = int(df_taskset.loc[index, 'TASK3_ID'])
            fourth_task = int(df_taskset.loc[index, 'TASK4_ID'])
            tasks = []

            if first_task != -1:

                task_info = df_task.loc[df_task['Task_ID'] == first_task]
                tasks.append(int(task_info['Priority']))
                tasks.append(int(task_info['Period'] / 1000))
                tasks.append(int(task_info['Number_of_Jobs']))
                n = str(task_info['PKG'].item())
                tasks.append(PKGs[n])
                tasks.append(int(task_info['Arg']))
                tasks.append(int(task_info['CRITICALTIME'] / 1000))
                job_info = df_job.loc[(df_job['Task_ID'] == first_task) & (df_job['Set_ID'] == grid)]
                for ind, r in job_info.iterrows():
                    tasks.append(Exit_Values[job_info.loc[ind, 'Exit_Value']])

            if second_task != -1:
                first_task = second_task
                task_info = df_task.loc[df_task['Task_ID'] == first_task]
                tasks.append(int(task_info['Priority']))
                tasks.append(int(task_info['Period'] / 1000))
                tasks.append(int(task_info['Number_of_Jobs']))
                n = str(task_info['PKG'].item())
                tasks.append(PKGs[n])
                tasks.append(int(task_info['Arg']))
                tasks.append(int(task_info['CRITICALTIME'] / 1000))
                job_info = df_job.loc[(df_job['Task_ID'] == first_task) & (df_job['Set_ID'] == grid)]
                for ind, r in job_info.iterrows():
                    tasks.append(Exit_Values[job_info.loc[ind, 'Exit_Value']])

            if third_task != -1:
                first_task = third_task
                task_info = df_task.loc[df_task['Task_ID'] == first_task]
                tasks.append(int(task_info['Priority']))
                tasks.append(int(task_info['Period'] / 1000))
                tasks.append(int(task_info['Number_of_Jobs']))
                n = str(task_info['PKG'].item())
                tasks.append(PKGs[n])
                tasks.append(int(task_info['Arg']))
                tasks.append( int ( task_info['CRITICALTIME'] / 1000))
                job_info = df_job.loc[(df_job['Task_ID'] == first_task) & (df_job['Set_ID'] == grid)]
                for ind, r in job_info.iterrows():
                    tasks.append(Exit_Values[job_info.loc[ind, 'Exit_Value']])

            if fourth_task != -1:
                first_task = fourth_task
                task_info = df_task.loc[df_task['Task_ID'] == first_task]
                tasks.append(int(task_info['Priority']))
                tasks.append(int(task_info['Period']))
                tasks.append(int(task_info['Number_of_Jobs']))
                n = str(task_info['PKG'].item())
                tasks.append(PKGs[n])
                tasks.append(int(task_info['Arg']))
                tasks.append(int(task_info['CRITICALTIME']))
                job_info = df_job.loc[(df_job['Task_ID'] == first_task) & (df_job['Set_ID'] == grid)]
                for ind, r in job_info.iterrows():
                    tasks.append(Exit_Values[job_info.loc[ind, 'Exit_Value']])
                labels = np.array(int(df_taskset.loc[index, 'Successful']))

            tasks = np.array(tasks)
            features.append(tasks)
            labels.append(res)
        except Exception as e:
            print(e)
            pass
        pbar.update(1)

labels = np.array(labels)
features = pad_sequences(features, maxlen=42, value=-1, padding='post', truncating='post')

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
