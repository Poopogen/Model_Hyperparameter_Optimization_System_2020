import numpy as np
import pandas as pd
from sklearn.preprocessing import scale,MinMaxScaler,StandardScaler,Normalizer,MaxAbsScaler


def do_normalize(data_train,data_test,Normalization_method):
    if Normalization_method=='StandardScaler':
        columns=data_train.columns 
        indexs_train = data_train.index 
        indexs_test = data_test.index


        scaler = StandardScaler()
        scaler = scaler.fit(data_train) 
        data_train_norm = scaler.transform(data_train) 
        data_train_norm = pd.DataFrame(data_train_norm,index=indexs_train,columns=columns)
        
        mean_train=scaler.mean_
        std_train=scaler.scale_
        df_mean_train=pd.DataFrame(mean_train,index=columns,columns=['Mean'])
        df_std_train=pd.DataFrame(std_train,index=columns,columns=['STDEV.P'])
        df_scaler_parameter = pd.concat([df_mean_train,df_std_train],axis=1)    
        df_scaler_parameter.to_csv('./Output_files/Normalization/scaler_parameter/parameter_StandardScaler.csv')
        

        scaled_index=df_scaler_parameter.index
        data_test_norm=data_test 
        for i in range(len(scaled_index)):
            #print(scaled_index[i])
            file_mean=df_scaler_parameter.loc[scaled_index[i]]['Mean']
            file_std=df_scaler_parameter.loc[scaled_index[i]]['STDEV.P']
            data_test_norm = data_test_norm.apply(lambda x: (x - file_mean) / file_std if x.name == '%s'%(scaled_index[i]) else x)
            

        data_train_norm.to_csv('./Output_files/Normalization/data_norm_stdscaler_train.csv')
        data_test_norm.to_csv('./Output_files/Normalization/data_norm_stdscaler_test.csv')
    
    
        return data_train_norm, data_test_norm

    if Normalization_method=='MinMaxScaler':
        columns=data_train.columns 
        indexs_train = data_train.index  
        indexs_test = data_test.index


        scaler = MinMaxScaler(feature_range=(0, 1))  
        scaler = scaler.fit(data_train) 
        data_train_norm = scaler.transform(data_train) 
        data_train_norm = pd.DataFrame(data_train_norm,index=indexs_train,columns=columns)
        
        max_train = scaler.data_max_
        min_train=scaler.data_min_
        df_max_train=pd.DataFrame(max_train,index=columns,columns=['Max'])
        df_min_train=pd.DataFrame(min_train,index=columns,columns=['Min'])
        df_scaler_parameter = pd.concat([df_max_train,df_min_train],axis=1)    
        df_scaler_parameter.to_csv('./Output_files/Normalization/scaler_parameter/parameter_MinMaxScaler.csv')
        

        scaled_index=df_scaler_parameter.index
        data_test_norm=data_test 
        for i in range(len(scaled_index)):
            #print(scaled_index[i])
            file_max=df_scaler_parameter.loc[scaled_index[i]]['Max']
            file_min=df_scaler_parameter.loc[scaled_index[i]]['Min']
            data_test_norm = data_test_norm.apply(lambda x: (x - file_min) / (file_max-file_min) if x.name == '%s'%(scaled_index[i]) else x)
            

        data_train_norm.to_csv('./Output_files/Normalization/data_norm_minmax_train.csv')
        data_test_norm.to_csv('./Output_files/Normalization/data_norm_minmax_test.csv')
        
        return data_train_norm, data_test_norm
    
    


def do_DeNormalize(result_data,Normalization_method):
    
    if Normalization_method=='StandardScaler':    
        df= pd.read_csv('./Output_files/Normalization/scaler_parameter/parameter_StandardScaler.csv',index_col=0,header=0)
        mean=df['Mean'][0] 
        std=df['STDEV.P'][0]
        denorm=result_data.apply(lambda x: (x*std)+ mean)
        #print('denorm_test:\n',denorm)

    if Normalization_method=='MinMaxScaler':
        df= pd.read_csv('./Output_files/Normalization/scaler_parameter/parameter_MinMaxScaler.csv',index_col=0,header=0)
        max_value=df['Max'][0] 
        min_value=df['Min'][0]
        denorm=result_data.apply(lambda x: (x*(max_value-min_value))+ min_value)
        #print('denorm_test:\n',denorm)

    return denorm


def shuffle(X,Y):
    np.random.seed()
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]
    

