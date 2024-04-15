#%%
import numpy as np
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC' 
config.gpu_options.per_process_gpu_memory_fraction = 0.99 
config.gpu_options.allow_growth =True 
set_session(tf.compat.v1.Session(config=config)) 
#from keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

import norm_denorm_shuffle as Norm_Denorm_Shuffle
import utilise as util
import buildData as BuildData

import shap





def pre_do_normalize(norm_file,label_file,data_test,Normalization_method):

    data_test_norm=data_test

    target_type=data_test_norm.columns.to_list()[0]

    
    if Normalization_method=='StandardScaler':
        df_scaler_parameter = pd.read_csv(norm_file,index_col=0,header=0)
        scaled_index=df_scaler_parameter.index
        
        for i in range(len(scaled_index)):
            #print(scaled_index[i])
            file_mean=df_scaler_parameter.loc[scaled_index[i]]['Mean']
            file_std=df_scaler_parameter.loc[scaled_index[i]]['STDEV.P']
            data_test_norm = data_test_norm.apply(lambda x: (x - file_mean) / file_std if x.name == '%s'%(scaled_index[i]) else x)
            

        data_test_norm.to_csv('./Prediction/data_norm_stdscaler_predicttest.csv')
    
    
        return data_test_norm,target_type

    if Normalization_method=='MinMaxScaler':#haven't modified yet
        df_scaler_parameter = pd.read_csv(norm_file,index_col=0,header=0)
        
        scaled_index=df_scaler_parameter.index
        data_test_norm=data_test 
        for i in range(len(scaled_index)):
            #print(scaled_index[i])
            file_max=df_scaler_parameter.loc[scaled_index[i]]['Max']
            file_min=df_scaler_parameter.loc[scaled_index[i]]['Min']
            data_test_norm = data_test_norm.apply(lambda x: (x - file_min) / (file_max-file_min) if x.name == '%s'%(scaled_index[i]) else x)
            

        
        data_test_norm.to_csv('./Prediction/data_norm_minmax_predicttest.csv')
        
        return data_test_norm,target_type


def pre_do_DeNormalize(norm_file,label_file,result_data,Normalization_method):

    df_scaler_parameter = pd.read_csv(norm_file,index_col=0,header=0)
    
    if Normalization_method=='StandardScaler':    
        #print('df_scaler_parameter:',df_scaler_parameter)
        mean=df_scaler_parameter['Mean'][0] 
        std=df_scaler_parameter['STDEV.P'][0]
        denorm=result_data.apply(lambda x: (x*std)+ mean)
        #print('denorm_test:\n',denorm)

    if Normalization_method=='MinMaxScaler':
        #df= pd.read_csv('./Output_files/Normalization/scaler_parameter/parameter_MinMaxScaler.csv',index_col=0,header=0)
        max_value=df_scaler_parameter['Max'][0] 
        min_value=df_scaler_parameter['Min'][0]
        denorm=result_data.apply(lambda x: (x*(max_value-min_value))+ min_value)
        #print('denorm_test:\n',denorm)

    return denorm


#==============================


model_path_input='./Prediction/Input/ModelSaved/'
model_path= os.listdir( model_path_input )
model_path.sort() 
print('Model File:\n',model_path)
if len(model_path) > 1 or len(model_path)==0:
    print('please check model file!!!')
else:    
    for file in model_path:
        model_file = os.path.join(model_path_input, file) 

norm_path_input='./Prediction/Input/Normalization_scaler_parameter/'
norm_path= os.listdir( norm_path_input )
norm_path.sort() 
print('Normalization File:\n',norm_path)
if len(norm_path) > 1 or len(norm_path)==0:
    print('please check normalization file!!!')
else:
    for file in norm_path:
        norm_file = os.path.join(norm_path_input, file) 


data_path_input='./Prediction/Input/InputData/'
inputdata_path= os.listdir( data_path_input )
inputdata_path.sort() 
print('Input Data File:\n',inputdata_path)
if len(inputdata_path) > 1 or len(inputdata_path)==0:
    print('please check data file!!!')
else:    
    for file in inputdata_path:
        data_file = os.path.join(data_path_input, file) 


label_file = './Prediction/Input/Label/label_aspentotag_input.csv'

prediction_report = dict(Predict_Data_file=str(inputdata_path),Model_file=str(model_path))

#rounds=1
way='LSTM'
do_normalization=True
do_denormalization=True
normalization_method='StandardScaler'
predict_lstm_builddata_method='BlocktoSlide' #what method the  input data for prediction==> from Block to Slide
pastDay=15
futureDay=1

data = pd.read_csv(data_file,index_col=0,header=0)
loaded_model = tf.keras.models.load_model(model_file)




if do_normalization==True:
    prediction_report["normalization_method"] = normalization_method
    test,target_type = pre_do_normalize(norm_file,label_file,data,normalization_method)

prediction_report["Predict_target"] = target_type


if way=='LSTM': 
    X_test,Y_test,Y_date_test=BuildData.builddata(test,pastDay=pastDay,futureDay=futureDay,BuildData_way=predict_lstm_builddata_method) #Y_date_train沒作用
if way=='NN':
    X_test,Y_test,Y_date_test=BuildData.NNbuilddata(test) 




predictions=loaded_model.predict(X_test)
print('\npredictions.shape:',predictions.shape)
prediction_report["Test data (Prediction)"] = predictions.shape[0]



df_pred = pd.DataFrame(predictions,columns=["Prediction"],index=Y_date_test)
df_real =pd.DataFrame(Y_test,columns=["Real"],index=Y_date_test)

df_result=pd.concat([df_real,df_pred],axis=1)

                

if do_denormalization:
    prediction_report["do_denormalization"] = 'yes'
    #df_result.to_csv("./Output_files/lstmself_NORMresult_bs{}_ep{}_{}layer{}_d{}_lr{}_{}.csv".format(self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr,self._round_no),index=Y_date_test) #just for save file

    df_result_DeNorm = pre_do_DeNormalize(norm_file,label_file,df_result, normalization_method) #mean_train, std_train
    
    
    plt.figure()#figsize=(w,h)
    df_result_DeNorm.plot()#xticks = df_result_DeNorm_plt[0]
    plt.title("Model Prediction_DeNormalization", fontsize=12)
    plt.xticks(rotation=15, fontsize=10)
    plt.savefig("./Prediction/output/predictplot_{}.png".format(str(model_path)[2:-5]))
    plt.show()

    print('\nNormalized data:')
    #print('%s: %.3f' % ('predict MSE',mean_squared_error(Y_test,predictions)))
    print('%s: %.3f' % ('RMSE',np.sqrt(mean_squared_error(Y_test,predictions))))


    print('\nDenormalized data:')
    #print('%s: %.3f' % ('predict MSE',mean_squared_error(df_result_DeNorm['Real'],df_result_DeNorm['Prediction'])))
    print('%s: %.3f' % ('RMSE',np.sqrt(mean_squared_error(df_result_DeNorm['Real'],df_result_DeNorm['Prediction']))))
    
    
    MAPE,MAE=util.do_mape_mae(df_result_DeNorm)
    print('%s: %.2f' % ('MAPE',MAPE),'%')
    print('%s: %.3f' % ('MAE',MAE))

    correlation=util.do_correlation(df_result_DeNorm)
    print('correlation:',correlation)
    

    RMSE=np.sqrt(mean_squared_error(df_result_DeNorm['Real'],df_result_DeNorm['Prediction']))
    CV=util.do_CV_calculation(df_result_DeNorm,RMSE)
    print('%s: %.2f' % ('CV',CV),'%')
        
    correct_direction=util.do_correct_direction(df_result_DeNorm)
    print('%s: %.1f' % ('correct_direction',correct_direction),'%')
    
    predict_result=df_result_DeNorm
else: 
    #df_result.to_csv("./Output_files/lstmself_result_bs{}_ep{}_{}layer{}_d{}_lr{}.xlsx".format(self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr),index=Y_date_test)

    #df_result_csv = pd.read_csv("./Output_files/lstmself_result_bs{}_ep{}_{}layer{}_d{}_lr{}.xlsx".format(self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr),index_col=0,header=0)
    plt.figure()
    df_result.plot()#xticks = df_result_plt[0]
    plt.title("Model Prediction", fontsize=12)
    plt.xticks(rotation=15, fontsize=10)
    plt.savefig("./Prediction/output/predictplot_{}.png".format(str(model_path)[2:-5]))
    plt.show()


    MSE=mean_squared_error(Y_test,predictions)
    RMSE=np.sqrt(mean_squared_error(Y_test,predictions))
    #print('\n%s: %.2f' % ('predict MSE',MSE))
    print('\n%s: %.2f' % ('predict RMSE',RMSE))

    correlation=util.do_correlation(df_result)
    print('%s: %.3f' % ('correlation',correlation))

    MAPE,MAE=util.do_mape_mae(df_result)
    print('%s: %.2f' % ('MAPE',MAPE),'%')
    print('%s: %.3f' % ('MAE',MAE))

    CV=util.do_CV_calculation(df_result,RMSE)
    print('%s: %.2f' % ('CV',CV),'%')


    correct_direction=util.do_correct_direction(df_result)
    print('%s: %.1f' % ('correct_direction',correct_direction),'%')

    predict_result=df_result
    #predict_round='Prediction_%s'%self._round_no

    #return predict_result,RMSE,MAE,MAPE,correlation,correct_direction,CV#,predict_round
prediction_report['RMSE']=RMSE
prediction_report['MAE']=MAE
prediction_report['MAPE(%)']=MAPE
prediction_report['Correlation']=correlation
prediction_report['Correct_Direction(%)']=correct_direction
prediction_report['CV(%)']=CV

prediction_report=pd.DataFrame.from_dict(prediction_report, orient='index')
print(str(model_path)[2:-5])
with pd.ExcelWriter("./Prediction/output/predictplot_{}.xlsx".format(str(model_path)[2:-5])) as writer:  
    prediction_report.to_excel(writer, sheet_name='Prediction_Report')
    predict_result.to_excel(writer, sheet_name='Prediction_Result')



#==========SHAP==========#
rounds=5

#Train Settings:
way='LSTM'
do_normalization=True
normalization_method='StandardScaler'
lstm_builddata_method='Slide'
pastDay=15
futureDay=1
shap_train=pd.read_csv('./Prediction/shap/shapTrain/online_test_data_forshap.csv',index_col=0,header=0)




f_names = shap_train.columns[1:]
print(f_names)
print('shap_train.shape:',shap_train.shape)
if do_normalization==True:
    shap_train, nonusetest = Norm_Denorm_Shuffle.do_normalize(shap_train,test,normalization_method)


if way=='LSTM':
    shap_X_train,Y_train,Y_date_train=BuildData.builddata(shap_train,pastDay=pastDay,futureDay=futureDay,BuildData_way=lstm_builddata_method) #Y_date_train沒作用
if way=='NN':
    shap_X_train,Y_train,Y_date_train=BuildData.NNbuilddata(shap_train)


for i in range(rounds):
    print('round_no:',i)
    background=shap_X_train[np.random.choice(shap_X_train.shape[0], 2000, replace=False)]
    print('shap_train.shape:\n',background.shape)

    explainer = shap.DeepExplainer(loaded_model,background)
    #X_test=X_test[np.random.choice(X_test.shape[0], 150, replace=False)]

    shap_values = explainer.shap_values(X_test)

    shap_val = np.array(shap_values)
    print('shap_val.shape:\n',shap_val.shape)
    shap_val = np.reshape(shap_val,(int(shap_val.shape[1]),int(shap_val.shape[2]),int(shap_val.shape[3])))
    print('shap_val.shape2:\n',shap_val.shape)
    shap_abs = np.absolute(shap_val)
    sum_0 = np.sum(shap_abs,axis=0)
    sum_total = np.sum(sum_0,axis=0)

    x_pos = np.array([i for i, _ in enumerate(f_names)])
    #print(x_pos.shape)
    print('sum_0.shape:\n',sum_0.shape)
    print('sum_0[0].shape:\n',sum_0[0].shape)
    print('sum_total.shape:\n',sum_total.shape)
    print('sum_total[0].shape:\n',sum_total.shape)

    if i==0:
        avgcal_sum_0 = sum_0
    else: 
        avgcal_sum_0 = avgcal_sum_0 + sum_0

avgcal_sum_0 = avgcal_sum_0/rounds
avgcal_sum_total = np.sum(avgcal_sum_0,axis=0)
print('avgcal_sum_total.shape:\n',avgcal_sum_total.shape)


CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
colorlist=['#2CBDFE','#F5B14C','#47DBCD','#F3A0F2','#A43512','#E2AE9A','#B1969F','#848586','#4A5C78','#661D98','#A37867','#9A75A1']#,'#E8EFE6'

fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(4,1,1)
#plot1_lines=ax.barh(x_pos,sum_0[0],tick_label=f_names)
plot1_lines=ax.barh(x_pos,avgcal_sum_0[0],tick_label=f_names)
ax.set_title('timestep 1')
ax.xaxis.label.set_size(12)
for i in range(len(f_names)):
    plot1_lines[i].set_color(colorlist[i])

ax2 = fig.add_subplot(4,1,2,sharex=ax)
#plot2_lines=ax2.barh(x_pos,sum_0[39],tick_label=f_names)
plot2_lines=ax2.barh(x_pos,avgcal_sum_0[3],tick_label=f_names)#39
ax2.set_title('timestep 4')#40
ax2.xaxis.label.set_size(12)
for i in range(len(f_names)):
    plot2_lines[i].set_color(colorlist[i])

ax3 = fig.add_subplot(4,1,3,sharex=ax)
#plot3_lines=ax3.barh(x_pos,sum_0[79],tick_label=f_names)
plot3_lines=ax3.barh(x_pos,avgcal_sum_0[7],tick_label=f_names)#79
ax3.set_title('timestep 8')#80
ax3.xaxis.label.set_size(12)
for i in range(len(f_names)):
    plot3_lines[i].set_color(colorlist[i])

ax4 = fig.add_subplot(4,1,4,sharex=ax)
#plot4_lines=ax4.barh(x_pos,sum_0[119],tick_label=f_names)
plot4_lines=ax4.barh(x_pos,avgcal_sum_0[14],tick_label=f_names)#119
ax4.set_title('timestep 15')#120
ax4.xaxis.label.set_size(12)
for i in range(len(f_names)):
    plot4_lines[i].set_color(colorlist[i])
plt.title('Shap Value')
plt.tight_layout()
plt.savefig('./Prediction/shap/timestep.png')
#plt.show()

fig , ax5 = plt.subplots()#figsize=(9, 7)
#plot5_lines=ax5.barh(x_pos,sum_total,tick_label=f_names)
plot5_lines=ax5.barh(x_pos,avgcal_sum_total,tick_label=f_names)
ax5.set_title('Overall Shap Value')
for i in range(len(f_names)):
    plot5_lines[i].set_color(colorlist[i])
plt.tight_layout()
fig.subplots_adjust(hspace=0.4, wspace=0.4)#,bottom=0.04,top=0.97
plt.savefig('./Prediction/shap/total_shapplot.png')


#==========Global Plots===============#
shap_values_2D = shap_values[0].reshape(-1,len(f_names))
X_test_2D = X_test.reshape(-1,len(f_names))
print("shap_values_2D.shape:",shap_values_2D.shape, "X_test_2D.shape:", X_test_2D.shape)
x_test_2d = pd.DataFrame(data=X_test_2D, columns = f_names)
x_test_2d.corr()

plt.figure(figsize=(13,10))  
shap.summary_plot(shap_values_2D, x_test_2d[0:60], show=False)
plt.tight_layout()
plt.savefig('./Prediction/shap/summary_plot.png')

plt.figure(figsize=(13,10))
shap.summary_plot(shap_values_2D, x_test_2d[0:60], plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('./Prediction/shap/summary_plot2.png')

print('SHAP Local Plots')
print('shap_values[0].shape:',shap_values[0].shape)
num=shap_values[0].shape[0]
shap.initjs()

for i in range(0,num):#sample_checked:
    print('sample',i+1)
    plt.figure(figsize=(15,20)) 
    sample_tstotal_shapevalues=np.sum(shap_values[0][i],axis=0)#shap_values[0][0]: (15, 8)
    shap.force_plot(explainer.expected_value[0], sample_tstotal_shapevalues, f_names, show=False, matplotlib=True)
    if i+1 in [1,3,5,6,8,10,11,13,15,16,17,19,21,23,25]:
        ##plt.tight_layout()
        plt.savefig('./Prediction/shap/localplot_sample%s.png'%str(i+1),bbox_inches = 'tight')#,dpi = 150
    #The base value is the average of all output values of the model on the training
print('end')
# %%
