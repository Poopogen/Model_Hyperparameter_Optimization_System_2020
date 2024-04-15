import numpy as np
import pandas as pd
import os
import utilise as util
import buildData as BuildData
import norm_denorm_shuffle as Norm_Denorm_Shuffle
#from check import check_gpumem 
#-------------------------------------------------------------------------------------------------------------------------#
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' 
config.gpu_options.per_process_gpu_memory_fraction = 0.50 
config.gpu_options.allow_growth =True 
set_session(tf.Session(config=config)) 
#-------------------------------------------------------------------------------------------------------------------------#

import keras
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Bidirectional
from keras.layers.normalization import BatchNormalization
#from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
#from keras.utils import plot_model


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



class Model:
    def __init__(self,model_use,data_train,data_test,file_list_train,file_list_test,normalization_method,train_size=0,batchsize=0,epoch=0,layer=0,unit=0,dropout=0,opt=0,do_shuffle=True,round_no=1):
        self._model_use=model_use
        self._training_report = dict(Train_file=str(file_list_train),Test_file=str(file_list_test))
        self._training_report["Model Type"]= model_use
        self._training_report["Parameters"]='batchsize{}_epoch{}_{}layer_unit{}_dropout{}_lr{}'.format(batchsize,epoch,layer,unit,dropout,opt)
        self._training_report["Train_size"]=train_size
        #self._training_report["Builddata_method"]=lstm_builddata_method
        self._training_report["Normalization_method"]=normalization_method


        self._data_train = data_train
        self._data_test = data_test

        if do_shuffle=='True':
            self._do_shuffle=True
        if do_shuffle=='False':
            self._do_shuffle=False

        #data settings
        self._normalization_method = str(normalization_method)

        self._train_size=float(train_size)
        self._batchsize =int(batchsize)
        
        self._model = Sequential()

        self._epoch=int(epoch)
        self._layer=int(layer)
        self._unit=unit 
        self._dropout=float(dropout)
        self._lr=float(opt)

        self._do_shuffle=do_shuffle
        
        self._round_no=str(round_no)

        
    def getTrainingReport(self):
        #print(self._training_report)
        return self._training_report


class ModelLSTM(Model):
    def __init__(self,model_use,data_train,data_test,file_list_train,file_list_test,normalization_method,train_size,batchsize,epoch,layer,unit,dropout,opt,do_shuffle,round_no,lstm_builddata_method,pastDay=0,futureDay=0):
        super().__init__(model_use,data_train,data_test,file_list_train,file_list_test,normalization_method,train_size,batchsize,epoch,layer,unit,dropout,opt,do_shuffle,round_no)

        self._training_report["Builddata_method"]=lstm_builddata_method

        self._lstm_builddata_method=str(lstm_builddata_method)

        if self._normalization_method == '/':
            if self._lstm_builddata_method=='Block':
                self._do_normalization=False
                self._do_denormalization=False
            if self._lstm_builddata_method=='Slide':
                self._do_normalization=False
                self._do_denormalization=False

        else:
            if self._lstm_builddata_method=='Block':
                self._do_normalization=True
                self._do_denormalization=False

            if self._lstm_builddata_method=='Slide':
                self._do_normalization=True
                self._do_denormalization=True


        self._pastDay=int(pastDay)    # MANY ("Many" to 1)  origintestfile=6 (115min)
        self._futureDay=int(futureDay)   # 1 (Many to "1")     origintestfile=1

    

    def train(self):
        if self._do_normalization==True:
            self._data_train, self._data_test = Norm_Denorm_Shuffle.do_normalize(self._data_train,self._data_test,self._normalization_method)
        
        X_data,Y_data,Y_date_train=BuildData.builddata(self._data_train,pastDay=self._pastDay,futureDay=self._futureDay,BuildData_way=self._lstm_builddata_method) #Y_date_train沒作用

        
        if self._do_shuffle==True:
            X_data,Y_data=Norm_Denorm_Shuffle.shuffle(X_data,Y_data)

        x_train, x_val, y_train, y_val = train_test_split(X_data, Y_data, train_size=self._train_size, shuffle=False) # shuffle=False (shuffle must be False here)

        print('x_train shape:', x_train.shape)
        print('x_val.shape:',x_val.shape)
        print('y_train.shape:',y_train.shape)
        print('y_val.shape:',y_val.shape)

        self._training_report["Training data"] = x_train.shape[0]
        self._training_report["Validating data"] = x_val.shape[0]
        self._training_report["Timesteps"] = x_train.shape[1]
        self._training_report["Features"] = x_train.shape[2]
        




        timesteps = x_train.shape[1] #10
        data_dim = x_train.shape[2] 

        if self._layer==1:
            self._model.add(LSTM(self._unit[0], batch_input_shape=(None, timesteps, data_dim),return_sequences=False))
            if self._dropout>0:
                self._model.add(Dropout(self._dropout))
        elif self._layer>1:  
            self._model.add(LSTM(self._unit[0], batch_input_shape=(None, timesteps, data_dim),return_sequences=True))    
            if self._dropout>0:
                self._model.add(Dropout(self._dropout))
            for n in range(self._layer - 2):
                self._model.add(LSTM(self._unit[n+1],return_sequences=True))
                if self._dropout>0:
                    self._model.add(Dropout(self._dropout))
            self._model.add(LSTM(self._unit[-1],return_sequences=False))
            if self._dropout>0:
                self._model.add(Dropout(self._dropout))
        else:
            print('please check the layer parameter(must be >=1).')
        self._model.add(Dense(1))        

        optimizer=optimizers.Adam(lr=self._lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self._model.compile(loss='mae', optimizer=optimizer,metrics=['mae', 'mean_absolute_percentage_error'])#mse
        self._model.summary()


        callback = EarlyStopping(monitor="loss", patience=100, verbose=1)


        #check_gpumem()

        history=self._model.fit(x_train, y_train, 
            epochs=self._epoch, 
            batch_size=self._batchsize,
            verbose = 2, 
            validation_data=(x_val, y_val), 
            callbacks=[callback],
            shuffle=True) 

        self._model.save('./Output_files/ModelSaved/model_{}_{}_ts{}_bs{}_ep{}_{}layer{}_d{}_lr{}_{}.h5'.format(self._model_use,self._lstm_builddata_method,self._pastDay,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr,self._round_no))
        #print(history.history.keys())  #dict_keys(['val_loss', 'loss'])


        # plot loss metric
        plt.figure(1)
        plt.plot(history.history['loss'], '-',label='train loss_R%s'%self._round_no)
        plt.plot(history.history['val_loss'], '--',label='validation loss_R%s'%self._round_no)
        plt.title('Train loss MSE per epoch')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        lg=plt.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
        plt.savefig("./Output_files/Plot/Loss_plot/mse/loss_per_epoch_{}_{}_ts{}_bs{}_ep{}_{}layer{}_d{}_lr{}_mse.png".format(self._model_use,self._lstm_builddata_method,self._pastDay,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr),
            bbox_extra_artists=(lg,), 
            bbox_inches='tight')
        plt.show()

        plt.figure(2)
        plt.plot(history.history['mean_absolute_error'], '-',label='train mae_R%s'%self._round_no)
        plt.plot(history.history['val_mean_absolute_error'], '--',label='validation mae_R%s'%self._round_no)
        plt.title('MAE per epoch')
        plt.ylabel('mae')
        plt.xlabel('epoch')
        #plt.legend(loc="upper right")
        lg=plt.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
        #plt.tight_layout()
        plt.savefig("./Output_files/Plot/Loss_plot/mae/loss_per_epoch_{}_{}_ts{}_bs{}_ep{}_{}layer{}_d{}_lr{}_mae.png".format(self._model_use,self._lstm_builddata_method,self._pastDay,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr),
            bbox_extra_artists=(lg,), 
            bbox_inches='tight')
        plt.show()

        plt.figure(3)
        plt.plot(history.history['mean_absolute_percentage_error'], '-',label='train mape_R%s'%self._round_no)
        plt.plot(history.history['val_mean_absolute_percentage_error'], '--',label='validation mape_R%s'%self._round_no)
        plt.title('MAPE per epoch')
        plt.ylabel('mape')
        plt.xlabel('epoch')
        lg=plt.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
        plt.savefig("./Output_files/Plot/Loss_plot/mape/loss_per_epoch_{}_{}_ts{}_bs{}_ep{}_{}layer{}_d{}_lr{}_mape.png".format(self._model_use,self._lstm_builddata_method,self._pastDay,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr),
            bbox_extra_artists=(lg,), 
            bbox_inches='tight')
        plt.show()


        print('\n\n\nevaluation processing....\n')
        evaluate_results =self._model.evaluate(x_val, y_val, batch_size=self._batchsize, verbose=1, sample_weight=None, steps=None)
        self._training_report["Evaluate_results(mse)"] = evaluate_results[0]
        self._training_report["Evaluate_results(mae)"] = evaluate_results[1]
        self._training_report["Evaluate_results(mape)"] = evaluate_results[2]
        print('\nval_loss(mse), val_mae, val_mape:', evaluate_results)
        #print("\n%s: %.3f" % ('val_loss(RMSE): ', pow(evaluate_results[0],0.5)))
        
            
    def predict(self):       

        print('\n\n\nprediction processing....\n')
        #when diff. buildmethod in both training(slide) and testing data(block)
        x_test,y_test,Y_date_test=BuildData.builddata(self._data_test,pastDay=self._pastDay,futureDay=self._futureDay,BuildData_way='Block')#self._lstm_builddata_method
        
        # when same buildmethod in both training and testing data
        # x_test,y_test,Y_date_test=BuildData.builddata(self._data_test,pastDay=self._pastDay,futureDay=self._futureDay,BuildData_way=self._lstm_builddata_method)#
        
        # print('\nY_date_test:\n',Y_date_test)
        # print('\ny_test:\n',y_test)
        # print('\nx_test:\n',x_test)
        predictions=self._model.predict(x_test, batch_size=self._batchsize, verbose=1, steps=None)
        print('\npredictions.shape:',predictions.shape)
        self._training_report["Test data (Prediction)"] = predictions.shape[0]

        #print('predictions:',predictions)


        df_pred = pd.DataFrame(predictions,columns=["Prediction"],index=Y_date_test)
        df_real =pd.DataFrame(y_test,columns=["Real"],index=Y_date_test)

        df_result=pd.concat([df_real,df_pred],axis=1)

                

        if self._do_denormalization:
            
            # df_result.to_csv("./Output_files/lstmself_NORMresult_bs{}_ep{}_{}layer{}_d{}_lr{}_{}.csv".format(self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr,self._round_no),index=Y_date_test) #just for save file
            
            #when using different buildmethod in both training(slide) and testing data(block)
            df_result["Prediction"]=Norm_Denorm_Shuffle.do_DeNormalize(df_result["Prediction"], self._normalization_method) #mean_train, std_train
            df_result_DeNorm = df_result            
            # df_result_DeNorm.to_csv("./Output_files/lstmself_DENORMresult_bs{}_ep{}_{}layer{}_d{}_lr{}_{}.csv".format(self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr,self._round_no),index=Y_date_test) #just for save file

            #when using same buildmethod in both training and testing data
            # df_result_DeNorm=Norm_Denorm_Shuffle.do_DeNormalize(df_result, self._normalization_method) #mean_train, std_train
            
            
            plt.figure()#figsize=(w,h)
            df_result_DeNorm.plot()#xticks = df_result_DeNorm_plt[0]
            plt.title("Model Prediction_DeNormalization", fontsize=12)
            plt.xticks(rotation=15, fontsize=10)
            plt.savefig("./Output_files/Plot/Prediction_plot/predictplot_{}_{}_ts{}_bs{}_ep{}_{}layer{}_d{}_lr{}_{}.png".format(self._model_use,self._lstm_builddata_method,self._pastDay,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr,self._round_no))
            plt.show()

            print('\nNormalized data:')
            #print('%s: %.3f' % ('predict MSE',mean_squared_error(y_test,predictions)))
            print('%s: %.3f' % ('RMSE',np.sqrt(mean_squared_error(y_test,predictions))))


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
            plt.savefig("./Output_files/Plot/Prediction_plot/predictplot_{}_{}_ts{}_bs{}_ep{}_{}layer{}_d{}_lr{}_{}.png".format(self._model_use,self._lstm_builddata_method,self._pastDay,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr,self._round_no))
            plt.show()


            MSE=mean_squared_error(y_test,predictions)
            RMSE=np.sqrt(mean_squared_error(y_test,predictions))
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

        return predict_result,RMSE,MAE,MAPE,correlation,correct_direction,CV#,predict_round



class ModelNN(Model):
    def __init__(self,model_use,data_train,data_test,file_list_train,file_list_test,normalization_method,train_size,batchsize,epoch,layer,unit,dropout,opt,do_shuffle,round_no):#,lstm_builddata_method,pastDay=0,futureDay=0
        super().__init__(model_use,data_train,data_test,file_list_train,file_list_test,normalization_method,train_size,batchsize,epoch,layer,unit,dropout,opt,do_shuffle,round_no)
        
        if self._normalization_method == '/':
            self._do_normalization=False
            self._do_denormalization=False
        else:
            self._do_normalization=True
            self._do_denormalization=True


    def train(self):
        if self._do_normalization==True:
            self._data_train, self._data_test = Norm_Denorm_Shuffle.do_normalize(self._data_train,self._data_test,self._normalization_method)
                
        X_data,Y_data,Y_date_train=BuildData.NNbuilddata(self._data_train)#Y_date_train沒作用
        
        if self._do_shuffle==True:
            X_data,Y_data=Norm_Denorm_Shuffle.shuffle(X_data,Y_data)


        x_train, x_val, y_train, y_val = train_test_split(X_data, Y_data, train_size=self._train_size, shuffle=False) # shuffle=False (shuffle must be False here)

        print('x_train shape:', x_train.shape)
        print('x_val.shape:',x_val.shape)
        print('y_train.shape:',y_train.shape)
        print('y_val.shape:',y_val.shape)

        self._training_report["Training data"] = x_train.shape[0]
        self._training_report["Validating data"] = x_val.shape[0]
        self._training_report["Timesteps"] = '/'
        self._training_report["Features"] = x_train.shape[1]
        

        feature_no = x_train.shape[1] 

        if self._layer==1:
            self._model.add(Dense(self._unit[0], input_shape=(feature_no,),activation='relu')) 
            # Now the model will take as input arrays of shape (None, feature_no) # and output arrays of shape (None, self._unit[0]).
            if self._dropout>0:
                self._model.add(Dropout(self._dropout))
        elif self._layer>1:  
            self._model.add(Dense(self._unit[0],activation='relu'))    
            if self._dropout>0:
                self._model.add(Dropout(self._dropout))
            for n in range(self._layer - 2):
                self._model.add(Dense(self._unit[n+1],activation='relu'))
                if self._dropout>0:
                    self._model.add(Dropout(self._dropout))
            self._model.add(Dense(self._unit[-1],activation='relu'))
            if self._dropout>0:
                self._model.add(Dropout(self._dropout))
        else:
            print('please check the layer parameter(must be >=1).')
        self._model.add(Dense(1))        

        optimizer=optimizers.Adam(lr=self._lr, beta_1=0.9, beta_2=0.999, amsgrad=False)#original:lr=0.001
        self._model.compile(loss='mae', optimizer=optimizer,metrics=['mae', 'mean_absolute_percentage_error'])#mse
        


        callback = EarlyStopping(monitor="loss", patience=100, verbose=1)



  
        history=self._model.fit(x_train, y_train, 
            epochs=self._epoch, 
            batch_size=self._batchsize,
            verbose = 2, 
            validation_data=(x_val, y_val), 
            callbacks=[callback],
            shuffle=True) 
            
        self._model.summary()
        self._model.save('./Output_files/ModelSaved/model_{}_bs{}_ep{}_{}layer{}_d{}_lr{}_{}.h5'.format(self._model_use,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr,self._round_no))

        #print(history.history.keys())  #dict_keys(['val_loss', 'loss'])


        # plot loss metric
        plt.figure(1)
        plt.plot(history.history['loss'], '-',label='train loss_R%s'%self._round_no)
        plt.plot(history.history['val_loss'], '--',label='validation loss_R%s'%self._round_no)
        plt.title('Train loss MSE per epoch')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        lg=plt.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
        plt.savefig("./Output_files/Plot/Loss_plot/mse/loss_per_epoch_{}_bs{}_ep{}_{}layer{}_d{}_lr{}_mse.png".format(self._model_use,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr),
            bbox_extra_artists=(lg,), 
            bbox_inches='tight')
        plt.show()

        plt.figure(2)
        plt.plot(history.history['mean_absolute_error'], '-',label='train mae_R%s'%self._round_no)
        plt.plot(history.history['val_mean_absolute_error'], '--',label='validation mae_R%s'%self._round_no)
        plt.title('MAE per epoch')
        plt.ylabel('mae')
        plt.xlabel('epoch')
        #plt.legend(loc="upper right")
        lg=plt.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
        #plt.tight_layout()
        plt.savefig("./Output_files/Plot/Loss_plot/mae/loss_per_epoch_{}_bs{}_ep{}_{}layer{}_d{}_lr{}_mae.png".format(self._model_use,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr),
            bbox_extra_artists=(lg,), 
            bbox_inches='tight')
        plt.show()

        plt.figure(3)
        plt.plot(history.history['mean_absolute_percentage_error'], '-',label='train mape_R%s'%self._round_no)
        plt.plot(history.history['val_mean_absolute_percentage_error'], '--',label='validation mape_R%s'%self._round_no)
        plt.title('MAPE per epoch')
        plt.ylabel('mape')
        plt.xlabel('epoch')
        lg=plt.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
        plt.savefig("./Output_files/Plot/Loss_plot/mape/loss_per_epoch_{}_bs{}_ep{}_{}layer{}_d{}_lr{}_mape.png".format(self._model_use,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr),
            bbox_extra_artists=(lg,), 
            bbox_inches='tight')
        plt.show()


        print('\n\n\nevaluation processing....\n')
        evaluate_results =self._model.evaluate(x_val, y_val, batch_size=self._batchsize, verbose=1, sample_weight=None, steps=None)
        self._training_report["Evaluate_results(mse)"] = evaluate_results[0]
        self._training_report["Evaluate_results(mae)"] = evaluate_results[1]
        self._training_report["Evaluate_results(mape)"] = evaluate_results[2]
        print('\nval_loss(mse), val_mae, val_mape:', evaluate_results)
        #print("\n%s: %.3f" % ('val_loss(RMSE): ', pow(evaluate_results[0],0.5)))
        


            
    def predict(self):       

        print('\n\n\nprediction processing....\n')
        x_test,y_test,Y_date_test=BuildData.NNbuilddata(self._data_test)
        #print('ytest:\n',y_test)
        predictions=self._model.predict(x_test, batch_size=self._batchsize, verbose=1, steps=None)
        print('\npredictions.shape:',predictions.shape)
        self._training_report["Test data (Prediction)"] = predictions.shape[0]

        #print('predictions:',predictions)


        df_pred = pd.DataFrame(predictions,columns=["Prediction"],index=Y_date_test)
        df_real =pd.DataFrame(y_test,columns=["Real"],index=Y_date_test)

        df_result=pd.concat([df_real,df_pred],axis=1)

                

        if self._do_denormalization:
            df_result.to_csv("./Output_files/lstmself_NORMresult_bs{}_ep{}_{}layer{}_d{}_lr{}_{}.csv".format(self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr,self._round_no),index=Y_date_test) #just for save file

            
            df_result_DeNorm=Norm_Denorm_Shuffle.do_DeNormalize(df_result, self._normalization_method) #mean_train, std_train
            plt.figure()#figsize=(w,h)
            df_result_DeNorm.plot()#xticks = df_result_DeNorm_plt[0]
            plt.title("Model Prediction_DeNormalization", fontsize=12)
            plt.xticks(rotation=15, fontsize=10)
            plt.savefig("./Output_files/Plot/Prediction_plot/predictplot_{}_bs{}_ep{}_{}layer{}_d{}_lr{}_{}.png".format(self._model_use,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr,self._round_no))
            plt.show()

            print('\nNormalized data:')
            #print('%s: %.3f' % ('predict MSE',mean_squared_error(y_test,predictions)))
            print('%s: %.3f' % ('RMSE',np.sqrt(mean_squared_error(y_test,predictions))))


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
            plt.savefig("./Output_files/Plot/Prediction_plot/predictplot_{}_bs{}_ep{}_{}layer{}_d{}_lr{}_{}.png".format(self._model_use,self._batchsize,self._epoch,self._layer,self._unit,self._dropout,self._lr,self._round_no))
            plt.show()


            MSE=mean_squared_error(y_test,predictions)
            RMSE=np.sqrt(mean_squared_error(y_test,predictions))
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

        return predict_result,RMSE,MAE,MAPE,correlation,correct_direction,CV#,predict_round

    
