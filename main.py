#%%
# # -*- coding: utf-8 -*-
#%matplotlib inline
#import numpy as np
import pandas as pd
import os
import sys


from buildModel import ModelLSTM,ModelNN

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#================================  READ FILE & FIXED SETTINGS======================================#
train_path_input = './Data/Train/'
test_path_input = './Data/Test/'

file_list_train= os.listdir( train_path_input )
file_list_train.sort() 
print('Train File:\n',file_list_train)

for file in file_list_train:
    file_train = os.path.join(train_path_input, file) 

data_train = pd.read_csv(file_train,index_col=0,header=0)


#read TEST file
file_list_test= os.listdir( test_path_input )
file_list_test.sort() 
print('\nTest File:\n',file_list_test)


for file in file_list_test:
    file_test = os.path.join(test_path_input, file) 

data_test = pd.read_csv(file_test,index_col=0,header=0)

#================================  TRAIN ======================================#
#print(sys.argv)
model_use=str(sys.argv[1])##
normalization_method=str(sys.argv[2])
lstm_builddata_method=str(sys.argv[3])
pastDay=sys.argv[4]
futureDay=sys.argv[5]
train_size=sys.argv[6]
batchsize=sys.argv[7]
epoch=sys.argv[8]


layer=list(map(int, sys.argv[9].split(',')))[0]
unit=list(map(int, sys.argv[9].split(',')))[1:]
opt=sys.argv[10]
dropout=sys.argv[11]
repeat_no=sys.argv[12]

print('\nmodel_use: {}\nnormalization_method: {}\nlstm_builddata_method: {}\npastDay(timesteps): {}\nfutureDay: {}\ntrain_size: {}\nbatchsize: {}\nepoch: {}\nlayer: {}\nunit: {}\ndropout: {}\nlearning_rate: {}\nrepeat_no: {}'.format(model_use,normalization_method,lstm_builddata_method,pastDay,futureDay,train_size,batchsize,epoch,layer,unit,dropout,opt,repeat_no))
predict_report = {'RMSE':[],'MAE':[],'MAPE(%)':[],'Correlation':[],'Correct_Direction(%)':[],'CV(%)':[]}

if len(unit)==layer:
   
   
    for n in range(int(repeat_no)):
        
        round_no=n+1
        print('\n\n\n=========== Round',round_no,'===========')

        if model_use=='LSTM':
            #model=ModelLSTM(model_use,data_train,data_test,file_list_train,file_list_test,normalization_method=normalization_method,lstm_builddata_method=lstm_builddata_method,pastDay=pastDay,futureDay=futureDay,train_size=train_size,batchsize=batchsize,epoch=epoch,layer=layer,unit=unit,dropout=dropout,opt=opt,do_shuffle=True,round_no=round_no)
            model=ModelLSTM(model_use,data_train,data_test,file_list_train,file_list_test,normalization_method,
                train_size=train_size,batchsize=batchsize,epoch=epoch,
                layer=layer,unit=unit,dropout=dropout,opt=opt,do_shuffle=True,round_no=round_no,
                lstm_builddata_method=lstm_builddata_method,pastDay=pastDay,futureDay=futureDay)

        if model_use=='NN':  
            model=ModelNN(model_use,data_train,data_test,file_list_train,file_list_test,normalization_method,
                train_size=train_size,batchsize=batchsize,epoch=epoch,
                layer=layer,unit=unit,dropout=dropout,opt=opt,do_shuffle=True,round_no=round_no)
        
        model.train()
        predict_result,RMSE,MAE,MAPE,correlation,correct_direction,CV=model.predict()
        predict_report['RMSE'].append(RMSE)
        predict_report['MAE'].append(MAE)
        predict_report['MAPE(%)'].append(MAPE)
        predict_report['Correlation'].append(correlation)
        predict_report['Correct_Direction(%)'].append(correct_direction)
        predict_report['CV(%)'].append(CV)

        train_report=model.getTrainingReport()
        train_report=pd.DataFrame.from_dict(train_report, orient='index')

        if model_use=='LSTM':
            with pd.ExcelWriter("./Output_files/Report/result_{}_{}_ts{}_bs{}_ep{}_{}layer{}_d{}_lr{}.xlsx".format(model_use,lstm_builddata_method,pastDay,batchsize,epoch,layer,unit,dropout,opt)) as writer:  
                train_report.to_excel(writer, sheet_name='Training_Report')
        if model_use=='NN':
            with pd.ExcelWriter("./Output_files/Report/result_{}_bs{}_ep{}_{}layer{}_d{}_lr{}.xlsx".format(model_use,batchsize,epoch,layer,unit,dropout,opt)) as writer:  
                train_report.to_excel(writer, sheet_name='Training_Report')

        if n==0:
            predict_result.rename(columns={'Prediction':'Prediction_%i'%round_no,}, 
                    inplace=True)
            predict_result_final=predict_result
        else:
            predict_result.rename(columns={'Prediction':'Prediction_%i'%round_no,}, 
                    inplace=True)
            predict_result_final=pd.concat([predict_result_final,predict_result],axis=1)
        #with pd.ExcelWriter("./Output_files/lstmself_result_bs{}_ep{}_{}layer{}_d{}_lr{}.xlsx".format(batchsize,epoch,layer,unit,dropout,opt),mode='a') as writer: 
            #predict_result.to_excel(writer, sheet_name='Prediction_{}'.format(round_no))

    predict_report=pd.DataFrame.from_dict(predict_report, orient='index')
    if model_use=='LSTM':
        with pd.ExcelWriter("./Output_files/Report/result_{}_{}_ts{}_bs{}_ep{}_{}layer{}_d{}_lr{}.xlsx".format(model_use,lstm_builddata_method,pastDay,batchsize,epoch,layer,unit,dropout,opt),mode='a') as writer:  
            predict_report.to_excel(writer, sheet_name='Prediction_Report')
            predict_result_final.to_excel(writer, sheet_name='Prediction_Result')

    if model_use=='NN':
        with pd.ExcelWriter("./Output_files/Report/result_{}_bs{}_ep{}_{}layer{}_d{}_lr{}.xlsx".format(model_use,batchsize,epoch,layer,unit,dropout,opt),mode='a') as writer:  
            predict_report.to_excel(writer, sheet_name='Prediction_Report')
            predict_result_final.to_excel(writer, sheet_name='Prediction_Result')


    print('\n\n============== END ===============')


else:
    print('Layer no. does not match unit settings.')




# %%
