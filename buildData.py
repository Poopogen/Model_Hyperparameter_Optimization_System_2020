import numpy as np
import pandas as pd
from numpy import newaxis



def builddata(data, pastDay, futureDay,BuildData_way): #pastDay=pastTime 6(25 mins) futureDay=futureTime?days
    X_data, Y_data = [], []
    Y_date = []


    if BuildData_way=='Slide':
        for i in range(data.shape[0]-futureDay-pastDay):
            X_data.append(np.array(data.iloc[i:i+pastDay , 1:]))
            Y_data.append(np.array(data.iloc[i+pastDay:i+pastDay+futureDay , 0]))  
            Y_date.extend(data.iloc[i+pastDay:i+pastDay+futureDay].index.values.tolist()) 


    if BuildData_way=='Block':

        data=data.reset_index() 
        
        X_data, Y_data = [], []
        Y_date = []
        futureDay=0
        

        for i in range(int(data.shape[0]/pastDay)):
            #print(i)
            block=i*pastDay
            #print('\nblock:',block)


            X_data.append(np.array(data.iloc[block:block+pastDay , 1:]))
            Y_data.append(np.array(data.iloc[block , 0]))   
            Y_date.extend(['sample %s'%str(i+1)])  
    
    if BuildData_way=='BlocktoSlide':
        
        data=data.reset_index() 
        
        X_data, Y_data = [], []
        Y_date = []
        futureDay=0
        

        for i in range(int(data.shape[0]/pastDay)):
            #print(i)
            block=i*pastDay
            #print('\nblock:',block)
            #print(data.iloc[block:block+pastDay , 2:])
            #print(data.iloc[block , 1])
            #print('=====')

            X_data.append(np.array(data.iloc[block:block+pastDay , 2:]))
            Y_data.append(np.array(data.iloc[block , 1])) 
            Y_date.append(data.iloc[block , 0])

            #Y_date.extend(['sample %s'%str(i+1)])  

    # print('\nY_date:\n',Y_date)
    # print('\nX_data:\n',X_data)
    # print('\nY_data:\n',Y_data)
    return np.array(X_data), np.array(Y_data), Y_date

def NNbuilddata(data):
    X=data.iloc[:, 1:]
    Y=data.iloc[:, 0]
    X_data=X.to_numpy().reshape((X.shape[0],X.shape[1]))#,1 #shape(batch_size,features)
    Y_data=Y.to_numpy().reshape((Y.shape[0],1))
    #print('X_data:\n',X_data)
    Y_date=data.index 
    #print(Y_date)
    return X_data,Y_data,Y_date


