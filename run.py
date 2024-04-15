from subprocess import call
from multiprocessing import Process
import nvidia_smi
#from waiting import wait
import time
import os
#import tqdm

model_use='LSTM'
gpus=1
normalization_method='StandardScaler'
lstm_builddata_method='Slide'
futureDay=1
train_size=0.9
repeat_no = 1


pastDays=[]
batchsizes=[]
epochs=[]
layers_units=[]
opts=[]
drops=[]



with open('./LSTM_List/timesteps', 'r') as f:
    for line in f:
        line = line.strip()
        pastDays.append(line)

with open('./LSTM_List/batchsize', 'r') as f:
    for line in f:
        line = line.strip()
        batchsizes.append(line)

with open('./LSTM_List/epoch', 'r') as f:
    for line in f:
        line = line.strip()
        epochs.append(line)

with open('./LSTM_List/layer_unit', 'r') as f:
    for line in f:
        line = line.strip()
        layers_units.append(line)


with open('./LSTM_List/lr', 'r') as f:
    for line in f:
        line = line.strip()
        opts.append(line)

with open('./LSTM_List/drop', 'r') as f:
    for line in f:
        line = line.strip()
        drops.append(line)

#print(pastDays,'\n',batchsizes,'\n',epochs,'\n',layers_units,'\n',opts,'\n',drops)

def check_gpumem(gpu_no):
    checklist=[]
    handle = []
    mem_res = []
    for i in range(gpu_no):
        nvidia_smi.nvmlInit()
        handle.append(nvidia_smi.nvmlDeviceGetHandleByIndex(i))
        #handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

        mem_res.append(nvidia_smi.nvmlDeviceGetMemoryInfo(handle[i]))
        #print(f'mem: {100 * (mem_res[i].used / mem_res[i].total):.3f}%') # percentage usage

        if (100 * (mem_res[i].used / mem_res[i].total))<1:
            x=True
        else:
            x=False
        checklist.append(x)
    return checklist

def gpu_ready(checklist):
    if any(gpu_checklist): #[False,False,False] is False; [False,True,False] is True
        return True
    return False

def worker(model_use,normalization_method,lstm_builddata_method,pastDay,futureDay,train_size,batchsize,epoch,layer_unit,opt,drop,repeat_no):
    call(["python", "main.py","%s"%model_use,"%s"%normalization_method,"%s"%lstm_builddata_method,pastDay,"%s"%futureDay,"%s"%train_size,"%s"%batchsize,"%s"%epoch,"%s"%layer_unit,"%s"%opt,"%s"%drop,"%s"%repeat_no ])#, ' >> /dev/null'
    


for layer_unit in layers_units:
    for batchsize in batchsizes:
        for epoch in epochs:
            for pastDay in pastDays:
                for opt in opts:
                    for drop in drops:
                        while True:
                            gpu_checklist=check_gpumem(gpus)
                            check_gpu=gpu_ready(gpu_checklist)
                            if check_gpu is True:
                                #wait(lambda: gpu_ready(gpu_checklist), timeout=120, waiting_for="waiting for gpu")
                                available_gpu=[i for i, x in enumerate(gpu_checklist) if x][0]
                                print('available_gpu = ', available_gpu)
                                print('check_gpu = ', check_gpu)
                                os.environ["CUDA_VISIBLE_DEVICES"]=str(available_gpu)
                                p = Process(target=worker, args=(model_use,normalization_method,lstm_builddata_method,pastDay,futureDay,train_size,batchsize,epoch,layer_unit,opt,drop,repeat_no))        
                                p.start()

                                time.sleep(10)
                                #call(["python", "main.py","%s"%model_use,"%s"%normalization_method,"%s"%lstm_builddata_method,pastDay,"%s"%futureDay,"%s"%train_size,"%s"%batchsize,"%s"%epoch,"%s"%layer_unit,"%s"%opt,"%s"%drop,"%s"%repeat_no ])
                                break
                        






