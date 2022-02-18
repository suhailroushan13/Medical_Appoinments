# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:41:17 2021

@author: J Venkat Reddy
"""
import os
module_dir = os.path.dirname(__file__)  # get current directory
file_path = os.path.join(module_dir,'KaggleV2-May-2016.csv')


def logistic():
    #importing the libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    #read the data
    global file_path
    df = pd.read_csv(file_path)

    # change columns name

    new_col_name = ['patient_id', 'appointment_id','gender','schedule_day','appointment_day','age','neighborhood',
                   'scholarship','hypertension','diabetes','alcoholism','handicap',
                   'sms_received','no_show']
    df.columns = new_col_name
    # check missing value
    df.isnull().sum()
    # change data type of some columns
    df['patient_id'] = df['patient_id'].astype('int64')
    df['schedule_day']= pd.to_datetime(df['schedule_day'])
    df['appointment_day']= pd.to_datetime(df['appointment_day'])


    df[df['age']< 0]
    # drop row with condition
    df.drop(df[df['age'] < 0].index, inplace =True)
    # make new column
    df['day'] = df.appointment_day.dt.day_name()
    ## drop columns
    df.drop(['patient_id', 'appointment_id','schedule_day',
             'appointment_day','neighborhood'], axis = 1, inplace = True)
    ## make separate coulumn
    showed_up = df[df.no_show == 'Showed up'].age
    not_showed_up = df[df.no_show != 'Showed up'].age

    df_3 = df.copy()
    # 1 for showed up, and 0 for not showed up
    df_3['no_show'] = df_3.no_show.map({'Yes':1, 'No':0})
    # create dummy variable and save this in df_1
    df_4 = pd.get_dummies(df_3, columns = ['gender','day'], drop_first = True)

    # import required libraries form scikit-learn
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score,precision_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    x=df_4.drop('no_show',axis=1)
    y=df_4['no_show'].values
    #splitting the datset into training and testing...
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
    #fit the traing data to the model...
    #logistic regression.................
    l=LogisticRegression()
    l.fit(x_train,y_train)
    y_pred=l.predict(x_test)
    cm=confusion_matrix(y_test,y_pred)
    accur=accuracy_score(y_test,y_pred)
    accur=round(accur,2)
    ps=round(precision_score(y_test,y_pred),1)
    print(accur,ps,"&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    return accur,ps
def d_tree():
    #importing the libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    #read the data
    global file_path
    df = pd.read_csv(file_path)

    # change columns name

    new_col_name = ['patient_id', 'appointment_id','gender','schedule_day','appointment_day','age','neighborhood',
                   'scholarship','hypertension','diabetes','alcoholism','handicap',
                   'sms_received','no_show']
    df.columns = new_col_name
    # check missing value
    df.isnull().sum()
    # change data type of some columns
    df['patient_id'] = df['patient_id'].astype('int64')
    df['schedule_day']= pd.to_datetime(df['schedule_day'])
    df['appointment_day']= pd.to_datetime(df['appointment_day'])


    df[df['age']< 0]
    # drop row with condition
    df.drop(df[df['age'] < 0].index, inplace =True)
    # make new column
    df['day'] = df.appointment_day.dt.day_name()
    ## drop columns
    df.drop(['patient_id', 'appointment_id','schedule_day',
             'appointment_day','neighborhood'], axis = 1, inplace = True)
    ## make separate coulumn
    showed_up = df[df.no_show == 'Showed up'].age
    not_showed_up = df[df.no_show != 'Showed up'].age

    df_3 = df.copy()
    # 1 for showed up, and 0 for not showed up
    df_3['no_show'] = df_3.no_show.map({'Yes':1, 'No':0})
    # create dummy variable and save this in df_1
    df_4 = pd.get_dummies(df_3, columns = ['gender','day'], drop_first = True)

    # import required libraries form scikit-learn
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score,precision_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    x=df_4.drop('no_show',axis=1)
    y=df_4['no_show'].values
    #splitting the datset into training and testing...
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
    l = DecisionTreeClassifier(criterion ="entropy",random_state=0)
    l.fit(x_train, y_train)
    y_pred = l.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    accur = accuracy_score(y_test, y_pred)
    accur = round(accur, 2)
    ps = round(precision_score(y_test, y_pred), 1)
    print(accur,ps,"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    return accur,ps
def forest():
    #importing the libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    #read the data
    global file_path
    df = pd.read_csv(file_path)

    # change columns name

    new_col_name = ['patient_id', 'appointment_id','gender','schedule_day','appointment_day','age','neighborhood',
                   'scholarship','hypertension','diabetes','alcoholism','handicap',
                   'sms_received','no_show']
    df.columns = new_col_name
    # check missing value
    df.isnull().sum()
    # change data type of some columns
    df['patient_id'] = df['patient_id'].astype('int64')
    df['schedule_day']= pd.to_datetime(df['schedule_day'])
    df['appointment_day']= pd.to_datetime(df['appointment_day'])


    df[df['age']< 0]
    # drop row with condition
    df.drop(df[df['age'] < 0].index, inplace =True)
    # make new column
    df['day'] = df.appointment_day.dt.day_name()
    ## drop columns
    df.drop(['patient_id', 'appointment_id','schedule_day',
             'appointment_day','neighborhood'], axis = 1, inplace = True)
    ## make separate coulumn
    showed_up = df[df.no_show == 'Showed up'].age
    not_showed_up = df[df.no_show != 'Showed up'].age

    df_3 = df.copy()
    # 1 for showed up, and 0 for not showed up
    df_3['no_show'] = df_3.no_show.map({'Yes':1, 'No':0})
    # create dummy variable and save this in df_1
    df_4 = pd.get_dummies(df_3, columns = ['gender','day'], drop_first = True)

    # import required libraries form scikit-learn
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score,precision_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    x=df_4.drop('no_show',axis=1)
    y=df_4['no_show'].values
    #splitting the datset into training and testing...
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
    classifier = RandomForestClassifier(n_estimators=20, criterion="entropy", random_state=0)
    classifier.fit(x_train, y_train)
    # predict..
    y_pred = classifier.predict(x_test)
    accur = accuracy_score(y_test, y_pred)
    accur = round(accur, 2)
    ps = round(precision_score(y_test, y_pred), 1)
    return accur,ps


def predict(a):
    #importing the libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    #read the data
    prediction_array = np.reshape(a, (-1, 13))
    global file_path
    df = pd.read_csv(file_path)

    # change columns name

    new_col_name = ['patient_id', 'appointment_id','gender','schedule_day','appointment_day','age','neighborhood',
                   'scholarship','hypertension','diabetes','alcoholism','handicap',
                   'sms_received','no_show']
    df.columns = new_col_name
    # check missing value
    df.isnull().sum()
    # change data type of some columns
    df['patient_id'] = df['patient_id'].astype('int64')
    df['schedule_day']= pd.to_datetime(df['schedule_day'])
    df['appointment_day']= pd.to_datetime(df['appointment_day'])


    df[df['age']< 0]
    # drop row with condition
    df.drop(df[df['age'] < 0].index, inplace =True)
    # make new column
    df['day'] = df.appointment_day.dt.day_name()
    ## drop columns
    df.drop(['patient_id', 'appointment_id','schedule_day',
             'appointment_day','neighborhood'], axis = 1, inplace = True)
    ## make separate coulumn
    showed_up = df[df.no_show == 'Showed up'].age
    not_showed_up = df[df.no_show != 'Showed up'].age

    df_3 = df.copy()
    # 1 for showed up, and 0 for not showed up
    df_3['no_show'] = df_3.no_show.map({'Yes':1, 'No':0})
    # create dummy variable and save this in df_1
    df_4 = pd.get_dummies(df_3, columns = ['gender','day'], drop_first = True)

    # import required libraries form scikit-learn
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score,precision_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    x=df_4.drop('no_show',axis=1)
    y=df_4['no_show'].values
    #splitting the datset into training and testing...
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
    classifier = RandomForestClassifier(n_estimators=20, criterion="entropy", random_state=0)
    classifier.fit(x_train, y_train)
    # predict..
    y_pred = classifier.predict(prediction_array)
    return y_pred








    # # a=[55,0,0,0,1,0,0,0,0,0,1,0,0]
    # # a=np.reshape(a, (-1,13))
    # # classifier.predict(a)
    # # df['day'].value_counts()
    # # return accur

