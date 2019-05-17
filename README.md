# KIA4SM_LSTM
This project intends to provide proof of concept to predict schedulability analysis using deep learning methods. A special type of RNN called LSTM is used.



Softwares and deep learning frameworks used:
* Python 3.7.2 
* cuDNN 7.5
* Keras 2.2.4 
* Pandas 0.24.2
* Scikit-learn 0.20.3 
* tqdm 4.31.01
* CUDA 10.1 
* Tensorflow-gpu 1.13.1
* graphviz 2.40

**1. Data selection and target variables_**
The first step is to preprocess the data. The database was imported and transformed into .csv files using DB Browser for SQLite (https://sqlitebrowser.org/)
 Only 6 features were selected from Tasks:
    1. Task Priority: Integer
    2. Task Period: Integer
    3. Task PKG: String
    4. Task Arg: Integer
    5. Task Critical time: Integer
    6. Number of Jobs: Integer
From Jobs only one feature was selected: Job Exit_Value: String.
After exporting all tables, start with Data_preparation.py. Line 165 is responsible for the length of the feature vector. 
Feature and labels are save in the end. 


**2. Training:**
CuDNNLSTM.py. When using CPU, install Tensorflow and replace CuDNNLSTM with LSTM

**3. Evaluation:**
Evaluation.py. Evaluation prints the confusion matrix and classification report. Tensorboard can be launched by typing tensorboard -–logdir=logs/ into the terminal and logs from trained models can be visualized 

**4. Prediction:**
predictin.py. A CSV file will be save with actual and predictied values. The trained model should be loaded first.

**5. Plotting:**
Plotting.py. Another way to visualize the model built.

**6. Optimization:**
parallel_search.py. 

