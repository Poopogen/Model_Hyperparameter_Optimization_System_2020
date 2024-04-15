import numpy as np
import pandas as pd
from sklearn import metrics
#import shap



def do_correlation(df_result):

#pearson correlation
    df_corr=df_result.corr(method='pearson') # return dataframe of correlation matrix
    corr=df_corr.loc['Prediction', 'Real']
    #print('corr:\n',corr)
    return corr

def do_mape_mae(df_result): #平均絕對百分比誤差(平均誤差百分比)MAPE #平均絕對誤差(平均誤差絕對值)MAE
    y_pred=df_result['Prediction']
    y_true=df_result['Real']
    mape_cal=np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    #print(mape_cal)
    mae_cal=metrics.mean_absolute_error(y_true, y_pred)
    #print(mae_cal)
    return mape_cal,mae_cal


def do_CV_calculation(df_result,rmse):
    df_real_mean=df_result['Real'].mean()
    
    CV=rmse/df_real_mean*100
    
    return CV






def do_correct_direction(data_result):
  
    data_predict=data_result['Prediction'].values
    data_predict_before=data_predict[:-1]
    data_predict_after=data_predict[1:]

    data_real=data_result['Real'].values
    data_real_before=data_real[:-1]
    data_real_after=data_real[1:]

    minus_real=data_real_before-data_real_after
    minus_predict=data_real_before-data_predict_after 

    filter_zero_predict= minus_predict==0
    filter_zero_real= minus_real==0


    part1 = filter_zero_real[filter_zero_predict==True]

    part2 = part1[part1 == True]
    bothzero_count = part2.sum()


    multiply=minus_real*minus_predict

    mult=multiply

    
    filter_negative = mult < 0 
    filter_positive = mult > 0 

    

    
    mult[filter_negative] = 0
    mult[filter_positive] = 1

    def _sum(arr,n): 
        # return sum using sum  
        # inbuilt sum() function 
        return(sum(arr))
    
    num=mult.shape[0]
    dosum=_sum(mult,num)
    dosum=dosum+bothzero_count
    
    correct_direction=(dosum/num)*100
   

    return correct_direction


# def do_cal_shap(model,trai_X_data,test_X_data):
# # SHAP model explainer
#     explainer = shap.DeepExplainer(model, train_X_data)
#     shap_value = explainer.shap_values(test_X_data)
#     shap_val = np.array(shap_value)
#     a = np.absolute(shap_val[0])
#     b = np.sum(a, axis=1)
#     SHAP_list = [np.sum(b[:, 0]), np.sum(b[:, 1]), np.sum(b[:, 2]), np.sum(b[:, 3]), np.sum(b[:, 4])]
#     #N_SHAP = normalize(SHAP_list)