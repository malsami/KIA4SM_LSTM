import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

# After exporting the relational database to separate tables with .csv extension, the transformation can begin
# The first step is to read the cvs files as Dataframes
df_taskset = pd.read_csv('TaskSet.csv') # import task-sets
# print(df_taskset.head()) if you want to see how the data look like

df_task = pd.read_csv('Task.csv')   # import tasks
# print(df_task.head())

df_job = pd.read_csv('Job.csv')  # import jobs
# print(df_job.head())


# 2. data transformation

# here starts data transformation
ntn = df_task[['PKG']].values  # get values from PKG in tasks. This step is equivalent to: Select distinct PKG from Task
ntn1 = []
for n in ntn:
    ntn1.append(n[0])
print(np.unique(ntn1)) # print the unique values



# PKG has a fixed set of labels. Integer encoding is used where integer # value is assigned to each label
PKGs = {}
PKGs['pi'] = 0
PKGs['hey'] = 1
PKGs['tumatmul'] = 2
PKGs['cond_mod'] = 3

# INteger encoding for Exit_Values from Jobs
Exit_Values = {}
Exit_Values['EXIT'] = 1
Exit_Values['EXIT_CRITICAL'] = 0


# ARG values ranged from 1 to 205.891.132.094.649, these values were normalized and scaled # to range from 1 to 17
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



# 3. Features and Labels extraction
i = 0

features = [] # create an empty list for features
labels = [] # create an empty list for labels
# loop in the task-set
with tqdm ( total=len (
        list(df_taskset.iterrows()))) as pbar:  # the total length would be total=len(list(df_taskset.iterrows()))
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
