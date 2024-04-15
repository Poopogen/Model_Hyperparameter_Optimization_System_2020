import os
import random
from scipy.stats import trapz

# epoch_range=range(3,10)
# epoch_term=3

layer_range=range(1,10) #must be>=1
layer_term=2


unit_range=range(1,500)
unit_multiplynum=16
unit_range = list(filter(lambda unit_range: unit_range%unit_multiplynum==0, unit_range))
unit_orderway='trapezoidal'  #'ascending','descending','trapezoidal','random'
unit_term=2


#===========================RANDOM CODE=============================#
#file_list= os.listdir( './List/' )
#print(file_list)
# with open('./List/epoch', 'r+') as epoch:
#     epoch.truncate(0)
#     epoch_random_list =sorted(random.sample(epoch_range,epoch_term))
#     print(epoch_random_list)
    
#     for n in range(epoch_term):
#         epoch_random=epoch_random_list[n]
#         #print(type(epoch_random))
#         #print(epoch_random)
#         epoch.write(str(epoch_random)+'\n')




with open('./List/layer_unit', 'r+') as unit:#LSTM_List#NN_List
    unit.truncate(0)
    layer_random_list =sorted(random.sample(layer_range,layer_term))
    #print('layer_random_list:',layer_random_list)
    for l in layer_random_list:
        #print('layer number in model:',l)
        for n in range(unit_term):
            #print('unit_term_id:',n)
            unit_random_create=[random.choice(unit_range) for _ in range(l)]#can repeat
            #print('Initial_unit_random_create:',unit_random_create)
            if unit_orderway=='ascending':
                unit_random_list =sorted(unit_random_create) #units in ascending order
            if unit_orderway=='descending':
                unit_random_list =sorted(unit_random_create,reverse=True) #units in decending order
            if unit_orderway=='trapezoidal':
                unit_range=list(unit_range)
                unit_random_list=[]
                for t,d in enumerate(range(len(unit_random_create))):
                    #print(t)
                    #print('unit_random_create:',unit_random_create)
                    if unit_random_create!=[]:   
                        m = max(unit_random_create)             
                        for i, j in enumerate(unit_random_create): 
                            if j == m :
                                if t%2==0:
                                    #print('t%2==0')
                                    unit_random_list=unit_random_list+[unit_random_create[i]]
                                    del unit_random_create[i]
                                else:
                                    #print('not t%2==0')
                                    unit_random_list=[unit_random_create[i]]+unit_random_list
                                    del unit_random_create[i]
                    else:
                        break               
            if unit_orderway=='random':
                unit_random_list=unit_random_create

            unit_random_list=[l]+unit_random_list
            unit.write(str(unit_random_list)[1:-1]+'\n')

            print('unit_random_list_sort:',unit_random_list)        





