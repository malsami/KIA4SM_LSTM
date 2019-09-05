from datetime import datetime
import pickle
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
import sqlite3

# After exporting the relational database to separate tables with .csv extension, the transformation can begin
# The first step is to read the cvs files as Dataframes
#df_taskset = pd.read_csv('TaskSet.csv') # import task-sets
# print(df_taskset.head()) if you want to see how the data look like

#df_task = pd.read_csv('Task.csv')   # import tasks
# print(df_task.head())

#df_job = pd.read_csv('Job.csv')  # import jobs
# print(df_job.head())


# PKG has a fixed set of labels. Integer encoding is used where integer # value is assigned to each label
PKGs = {
        'pi' : 0,
        'hey' : 1,
        'tumatmul' : 2,
        'cond_mod' : 3
        }

# Integer encoding for Exit_Values from Jobs
Exit_Values = {
        'EXIT' : 1,
        'EXIT_CRITICAL' : 0,
        'EXIT_PERIOD' : 2,
        'OUT_OF_CAPS' : 3,
        'OUT_OF_QUOTA' : 4
        }

# ARG values ranged from 1 to 205.891.132.094.649, these values were normalized and scaled # to range from 1 to 17
Arg_Values = {
        1 : 1,
        4096 : 2,
        8192 : 3,
        16384 : 4,
        32768 : 5,
        65536 : 6,
        131072 : 7,
        262144 : 8,
        524288 : 9,
        1048576 : 10,
        2097152 : 11,
        847288609443 : 12,
        2541865828329 : 13,
        7625597484987 : 14,
        22876792454961 : 15,
        68630377364883 : 16,
        205891132094649 : 17
        }


print("Doing writing")

DB_PATH = "/home/bernhard/panda_v4.db"
TASKS_DICT = {}

def taskToFeatureList(task):
    #returns a fature list for the corresponding task values
    feature = []
    feature.append(task['Priority'])
    feature.append(task['Period'])
    feature.append(task['Number_of_Jobs'])
    feature.append(task['PKG'])
    feature.append(task['Arg'])
    feature.append(task['CRITICALTIME'])
    return feature


def getTaskFeatures(db_path): #c is the cursor for the db
    # returns a dictionary 
    # { task_id : [ feature, list ]
    conn = sqlite3.connect(db_path)
    conn.row_factory = lambda C, R: { c[0]: R[i] for i, c in enumerate(C.description) }
    db_cursor  = conn.cursor()
    db_cursor.execute('select Task_ID,Priority,Period,PKG,Arg,CRITICALTIME,Number_of_Jobs from Task')
    outputTable  = db_cursor.fetchall()

    tasks_dict = {}
    for row in outputTable:
        row['Period'] = int(row['Period']/1000)
        row['Number_of_Jobs'] = int(row['Number_of_Jobs'])
        row['PKG'] = PKGs[row['PKG']]
        row['CRITICALTIME'] = int(row['CRITICALTIME']/1000)
        row['Arg'] = Arg_Values[row['Arg']]
        tasks_dict[row['Task_ID']] = taskToFeatureList(row)
    return tasks_dict


def processTaskset(tasksetData):
    # tasksetData is a list of tuples returned from the DB in getTasksetData()
    try:
        label = tasksetData[0][-1]
        features = []
        jobExitsByTask = {}
        for tsData in tasksetData:
            try:
                jobExitsByTask[tsData[4]].append(Exit_Values[tsData[5]])
            except KeyError:
                jobExitsByTask[tsData[4]] = [Exit_Values[tsData[5]]]
        for taskIdNo in (1,2,3):
            if tasksetData[0][taskIdNo] != -1:
                features += TASKS_DICT[tasksetData[0][taskIdNo]]
                features += jobExitsByTask[tasksetData[0][taskIdNo]]
    except KeyError as k:
        for t in tasksetData:
            print(t)
        raise k
    return np.array(features), label


def getFeaturesLabels(db_path):
    conn = sqlite3.connect(db_path)
    db_cursor = conn.cursor()
    command = 'SELECT TaskSet.Set_ID, TaskSet.TASK1_ID, TaskSet.TASK2_ID, TaskSet.TASK3_ID, Job.Task_ID, Job.Exit_Value, TaskSet.Successful'\
            ' FROM TaskSet JOIN Job'\
            ' ON TaskSet.Set_ID = Job.Set_ID and'\
            ' (TaskSet.TASK1_ID == Job.Task_ID or'\
            ' TaskSet.TASK2_ID == Job.Task_ID or'\
            ' TaskSet.TASK3_ID == Job.Task_ID);'
    db_cursor.execute(command)
    # data_table format: [( TaskSet.Set_ID, TaskSet.TASK1_ID, TaskSet.TASK2_ID, TaskSet.TASK3_ID, Job.Task_ID, Job.Exit_Value, TaskSet.Successful)]
    data_table = db_cursor.fetchall()
    print('reading taskset_jobs join done')
    print('The current time is:',datetime.now())
    finalFeatureList = []
    finalLabelList = []
    currentTset = data_table[0][0] # first taskset id
    tSetJobs = []
    totalSize = len(data_table)
    for row in data_table:
        if row[0] % 1000 == 0:
            print('processed',int(100 * (row[0]/totalSize)),'%' )
        if row[0] == currentTset:
            #then still same setTset
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
print('example task 222:',TASKS_DICT[222])

features, labels = getFeaturesLabels(DB_PATH)


print('The current time is:',datetime.now())

print("Done reading")
  
 
'''


# 2. data transformation

# here starts data transformation
#ntn = df_task[['PKG']].values  # get values from PKG in tasks. This step is equivalent to: Select distinct PKG from Task
#ntn1 = []
#for n in ntn:
#    ntn1.append(n[0])
#print(np.unique(ntn1)) # print the unique values









# 3. Features and Labels extraction
i = 0




sys.exit()
features = [] # create an empty list for features
labels = [] # create an empty list for labels
new_task_list = []
# loop in the task-set
with tqdm ( total=len( list(df_taskset.iterrows()))) as pbar:  # the total length would be total=len(list(df_taskset.iterrows()))

    for task_set in taskset_table:

            if task_set['TASK1_ID']!= -1:

                new_task_list.append()


                blub


    for index, row in df_taskset.iterrows ():

        try:

            i += 1
            grid = int(df_taskset.loc[index, 'Set_ID']) # task_set ID
            first_task = int(df_taskset.loc[index, 'TASK1_ID']) # first task_id
            second_task = int(df_taskset.loc[index, 'TASK2_ID']) # second task_id
            third_task = int(df_taskset.loc[index, 'TASK3_ID']) # third task_id
            fourth_task = int(df_taskset.loc[index, 'TASK4_ID']) # fourth task_id
            tasks = []  # empty list of tasks where features are saved later

            if first_task != -1: # if the first task exists in this task-set then :

                task_info = df_task.loc[df_task['Task_ID'] == first_task]
                tasks.append(int(task_info['Priority']))   # save the priority
                tasks.append(int(task_info['Period']/1000)) # save the period in seconds
                tasks.append(int(task_info['Number_of_Jobs'])) # save number of jobs
                n = str(task_info['PKG'].item())
                tasks.append(PKGs[n])             #save the numerical value of PKG
                av = int(task_info['Arg'].item())
                tasks.append(Arg_Values[av])    #save the scaled value of Arg
                tasks.append(int(task_info['CRITICALTIME']/1000))   # save criticaltime in seconds
                # for each job in that is in the task and has this task_set id
                job_info = df_job.loc[(df_job['Task_ID'] == first_task) & (df_job['Set_ID'] == grid)]

                for ind, r in job_info.iterrows():
                    tasks.append(Exit_Values[job_info.loc[ind, 'Exit_Value']])  # save the transformed exit value

            if second_task != -1:  # if the second task exists in this task-set then :
                first_task = second_task
                task_info = df_task.loc[df_task['Task_ID'] == first_task]
                tasks.append(int(task_info['Priority']))
                tasks.append(int(task_info['Period']/1000))
                tasks.append(int(task_info['Number_of_Jobs']))
                n = str(task_info['PKG'].item())
                tasks.append(PKGs[n])
                av = int(task_info['Arg'].item())
                tasks.append(Arg_Values[av])
                tasks.append(int(task_info['CRITICALTIME']/1000))
                print(tasks)
                job_info = df_job.loc[(df_job['Task_ID'] == first_task) & (df_job['Set_ID'] == grid)]
                for ind, r in job_info.iterrows():
                    tasks.append(Exit_Values[job_info.loc[ind, 'Exit_Value']])

            if third_task != -1:   # if the third task exists in this task-set then :
                first_task = third_task
                task_info = df_task.loc[df_task['Task_ID'] == first_task]
                tasks.append(int(task_info['Priority']))
                tasks.append(int(task_info['Period']/1000))
                tasks.append(int(task_info['Number_of_Jobs']))
                n = str(task_info['PKG'].item())
                tasks.append(PKGs[n])
                av = int(task_info['Arg'].item())
                tasks.append(Arg_Values[av])
                tasks.append(int(task_info['CRITICALTIME']/1000))

                job_info = df_job.loc[(df_job['Task_ID'] == first_task) & (df_job['Set_ID'] == grid)]
                for ind, r in job_info.iterrows():
                    tasks.append(Exit_Values[job_info.loc[ind, 'Exit_Value']])


            if fourth_task != -1: # if the fourth task exists in this task-set then :
                first_task = fourth_task
                task_info = df_task.loc[df_task['Task_ID'] == first_task]
                tasks.append(int(task_info['Priority']))
                tasks.append(int(task_info['Period']/1000))
                tasks.append(int(task_info['Number_of_Jobs']))
                n = str(task_info['PKG'].item())
                tasks.append(PKGs[n])
                av = int(task_info['Arg'].item())
                tasks.append(Arg_Values[av])
                tasks.append(int(task_info['CRITICALTIME']/1000))

                job_info = df_job.loc[(df_job['Task_ID'] == first_task) & (df_job['Set_ID'] == grid)]
                for ind, r in job_info.iterrows():
                    tasks.append(Exit_Values[job_info.loc[ind, 'Exit_Value']])


            tasks = np.array(tasks)  #  to save the task list as numpy array
            features.append(tasks)  # values in tasks are features
            labels.append(int(df_taskset.loc[index, 'Successful'])) # in the label list, append the value in the successful col from task-set
        except Exception as e:  # exception handler
            print(e)
            pass
        pbar.update(1)

'''

labels = np.array(labels) #  to save the labels list as numpy array

# To make a fixed length vector, if the vector is smaller than 56 then replace the empty values with -1. if longer than 56 trim the value
features = pad_sequences(features, maxlen=56, value=-1, padding='post', truncating='post')

#print(features.shape) # the dimensionality of features
#print(labels.shape) # the dimensionality of labels

#  save both files for the training
with open ( '56_features', 'wb' ) as outfile:  # 'wb' is the file mode, it means 'write binary'
    pickle.dump(features, outfile)

with open ( '56_labels', 'wb' ) as outfile:
    pickle.dump(labels, outfile)

print('The current time is:',datetime.now())
