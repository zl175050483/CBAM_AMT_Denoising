# main
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import scipy.io as sio
import math
from torch.utils.data import TensorDataset,DataLoader
from g15 import SCNet18

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SCNet18().to(device)

###parameters###
start = time.time()
EPOCH = 1000    
BATCH_SIZE_s = 100      
BATCH_SIZE_v = 1000
LR = 0.0001     
rate = 0.95  
iteration = 30 

###data loading###
dataset_s = 'dataset_sample.mat'
dataset_s = sio.loadmat(dataset_s)
dataset_s = dataset_s['dataset_sample'] 
#dataset_s = dataset_s[:,np.newaxis,:] 
Origin_s = dataset_s[0:10000,:,:]
MTsignal_s = dataset_s[10000:20000,:,:]
Ls=10000

dataset_v = 'dataset_val.mat'
dataset_v = sio.loadmat(dataset_v)
dataset_v = dataset_v['dataset_val'] 
#dataset_v = dataset_v[:,np.newaxis,:] 
Origin_v = dataset_v[0:1000,:,:]
MTsignal_v = dataset_v[1000:2000,:,:]
Lv=1000

###preprocessing###
def normalization(data,_range):
    return data / _range

matirx_s = np.row_stack((Origin_s,MTsignal_s))
range_s = np.max(abs(matirx_s))
norm_s = normalization(matirx_s,range_s)
np.save('range_s',range_s)
Origin_s = norm_s[0:Ls,:]
MTsignal_s = norm_s[Ls:Ls*2,:]

matirx_v = np.row_stack((Origin_v,MTsignal_v))
range_v = np.max(abs(matirx_v))
norm_v = normalization(matirx_v,range_v)
np.save('range_v',range_v)
Origin_v = norm_v[0:Lv,:]
MTsignal_v = norm_v[Lv:Lv*2,:]

x1_s = torch.from_numpy(Origin_s) 
x2_s = torch.from_numpy(MTsignal_s) 
x1_s = x1_s.type(torch.FloatTensor)
x2_s = x2_s.type(torch.FloatTensor)

x1_v = torch.from_numpy(Origin_v) 
x2_v = torch.from_numpy(MTsignal_v) 
x1_v = x1_v.type(torch.FloatTensor)
x2_v = x2_v.type(torch.FloatTensor)

train_data = TensorDataset(x2_s,x1_s)
val_data = TensorDataset(x2_v,x1_v)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_s, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE_v, shuffle=False, num_workers=0, drop_last=True)

###traning###
criterion = nn.MSELoss()  
criterion.cuda()
optimizer = optim.Adam(net.parameters(),lr=LR)
#optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

Losslist_s = []
Losslist_v = []
best_loss = 100
save_path = './net.pth'
print("Start Training!") 
for epoch in range(EPOCH):
    print('\nEpoch: %d' % (epoch + 1))    
    if epoch % iteration == 9:
        LR = LR*rate             
    loss_s = 0.0
    loss_v = 0.0
    #for i in range(Ls//BATCH_SIZE_s):
    for i, data_s in enumerate(train_loader, 0):
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        input_s, target_s = data_s
        input_s = input_s.to(device)
        output_s = net(input_s)
        target_s = target_s.to(device)
        loss_s0 = criterion(output_s, target_s)
        loss_s0.backward()
        optimizer.step()
        loss_s += loss_s0.item()
    Losslist_s.append(loss_s/(Ls//BATCH_SIZE_s))      
    net.eval()
    with torch.no_grad():
        #for j in range(Lv//BATCH_SIZE_v):
        for j, data_v in enumerate(val_loader, 0):
            input_v, target_v = data_v
            input_v = input_v.to(device)
            output_v = net(input_v)      
            target_v = target_v.to(device)
            loss_v0 = criterion(output_v, target_v) 
            loss_v += loss_v0.item()
        Losslist_v.append(loss_v/(Lv//BATCH_SIZE_v)) 
        if loss_v0 < best_loss:
            best_loss = loss_v0
            torch.save(net.state_dict(), save_path)
    if (epoch+1) % 1 == 0:
        print('train loss: {:.10f}'.format(loss_s/(Ls//BATCH_SIZE_s))) 
        print('val loss: {:.10f}'.format(loss_v/(Lv//BATCH_SIZE_v)))   
print('finished training')

###plot###
input_v = input_v.cpu()
input_v = input_v.detach().numpy()
output_v = output_v.cpu()
output_v = output_v.detach().numpy()
target_v = target_v.cpu()
target_v = target_v.detach().numpy()

x = range(1, EPOCH+1)
y_s = Losslist_s
y_v = Losslist_v
plt.semilogy(x, y_s, 'b.-')
plt.semilogy(x, y_v, 'r.-')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.show()
#plt.savefig("accuracy_loss.jpg")

#denoising change
col=10 #i-th denoising result
x = range(0, len(target_v[0,0,:]))
y1 = target_v[col,0,:]
y2 = input_v[col,0,:]
y3 = output_v[col,0,:]
plt.plot(x, y1, 'b.-')
plt.plot(x, y2, 'r.-')
plt.plot(x, y3, 'g.-')
plt.xlabel('Time')
plt.ylabel('Ampulitude')
plt.show()

###SNR###
# before denoising
origSignal = target_v
errorSignal = target_v-input_v
signal_2 = sum(origSignal.flatten()**2)
noise_2 = sum(errorSignal.flatten()**2)
SNRValues1 = 10*math.log10(signal_2/noise_2)
print(SNRValues1)

#after denoising
origSignal = target_v
errorSignal = target_v-output_v
signal_2 = sum(origSignal.flatten()**2)
noise_2 = sum(errorSignal.flatten()**2)
SNRValues2 = 10*math.log10(signal_2/noise_2)
print(SNRValues2)             
                    
end = time.time()
print (end-start)   

save_fn = 'y_s.mat'
save_array = y_s
sio.savemat(save_fn, {'y_s': save_array})

save_fn = 'y_v.mat'
save_array = y_v
sio.savemat(save_fn, {'y_v': save_array})
