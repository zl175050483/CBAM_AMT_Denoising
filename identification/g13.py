# main
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from g12 import SCNet18

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SCNet
net = SCNet18().to(device)

# parameters
EPOCH = 200   
BATCH_SIZE = 100      
LR = 0.0000008      

###loading###
load_fn1 = 'dataset_sample.mat'
load_data1 = sio.loadmat(load_fn1)
_dataset = load_data1['dataset_sample'] 
_dataset=_dataset

load_fn2 = 'dataset_val.mat'
load_data2 = sio.loadmat(load_fn2)
_dataset_val = load_data2['dataset_val'] 
_dataset_val=_dataset_val

###preprocessing###
def normalization(data):
    _range = np.max(abs(data))
    return data / _range

_data=normalization(_dataset)
_data_val=normalization(_dataset_val)

y1_data=np.zeros(10000)
y1_data[5000:10000]=1

x1_data=torch.from_numpy(_data) 
y1_data=torch.from_numpy(y1_data) 

x1_data = x1_data.type(torch.FloatTensor)
y1_data = y1_data.type(torch.FloatTensor)

y2_data=np.zeros(1000)
y2_data[500:1000]=1

x2_data=torch.from_numpy(_data_val) 
y2_data=torch.from_numpy(y2_data) 

x2_data = x2_data.type(torch.FloatTensor)
y2_data = y2_data.type(torch.FloatTensor)

deal_dataset1=TensorDataset(x1_data,y1_data)
trainloader=DataLoader(dataset=deal_dataset1,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

deal_dataset2=TensorDataset(x2_data,y2_data)
testloader=DataLoader(dataset=deal_dataset2,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

L0=len(x1_data[1,:])
L1=len(trainloader)
L2=len(testloader)
C1=len(trainloader.dataset)
C2=len(testloader.dataset)

classes = ('Noise-free', 'Noise')

criterion = nn.CrossEntropyLoss()  
optimizer=optim.Adam(net.parameters(),lr=LR)

# traning
sLoss_list = []
vLoss_list = []
sCorrect_list=[]
vCorrect_list=[]
best_correct = 0
save_path = './net.pth'

print("Start Training")  
for epoch in range(EPOCH):
    
    if epoch % 10 == 0:
        LR = LR*0.95
    
    net.train()
    s_loss=0.0
    v_loss=0.0
    s_correct = 0.0
    s_total = 0.0
    for i, data in enumerate(trainloader, 0):
        length = len(trainloader)
        inputs, labels = data
        #inputs = inputs.reshape(BATCH_SIZE,1,L0)
        inputs, labels = inputs.to(device), labels.to(device=device, dtype=torch.int64)
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        s_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        s_total += labels.size(0)
        s_correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
        % (epoch + 1, (i + 1 + epoch * length), s_loss / (i + 1), 100. * s_correct / s_total))
    sCorrect_list.append(100 * s_correct/C1)

        # each epoch's accuracy
    print("Waiting Test!")
    with torch.no_grad():
        v_correct = 0
        v_total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            #images = images.reshape(BATCH_SIZE,1,L0)
            images, labels = images.to(device), labels.to(device=device, dtype=torch.int64)
            outputs = net(images)
            val_loss = criterion(outputs, labels)
            v_loss += val_loss.item()
            #  (outputs.data index)
            _, predicted = torch.max(outputs.data, 1)
            v_total += labels.size(0)
            v_correct += (predicted == labels).sum()
        print('test identification accuracy：%.3f%%' % (100 * torch.true_divide(v_correct, v_total) )) #v_correct / v_total
        vCorrect_list.append(100 * torch.true_divide(v_correct, C2) )
        if v_correct > best_correct:
            best_correct = v_correct
            torch.save(net.state_dict(), save_path)
    
    sLoss_list.append(s_loss/L1)
    vLoss_list.append(v_loss/L2)
        
    if (epoch+1) % 1 == 0:
        print('train loss: {:.10f}'.format(s_loss/L1)) 
        print('val loss: {:.10f}'.format(v_loss/L2))  #length

print('finished training')

###plot###
x = range(1, EPOCH+1)
#y1 = np.array(sLoss_list)
#y2 = np.array(vLoss_list)

#y3 = np.array(sCorrect_list)
#y4 = np.array(vCorrect_list)

y1 = sLoss_list
y2 = vLoss_list

y3 = sCorrect_list
y4 = vCorrect_list
y4=torch.tensor(y4, device='cpu')

plt.subplot(2, 1, 1)
plt.plot(x, y1, 'b.-')
plt.plot(x, y2, 'r.-')
plt.title('Loss and Accuracy vs. Epochs')
plt.xlabel('Epoches')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(x, y3, 'bo-')
plt.plot(x, y4, 'ro-')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')

plt.show()
#plt.savefig("accuracy_loss.jpg")

save_fn = 'y1.mat'
save_array = y1
sio.savemat(save_fn, {'y1': save_array})

save_fn = 'y2.mat'
save_array = y2
sio.savemat(save_fn, {'y2': save_array})

y33 = np.array(y3)
save_fn = 'y33.mat'
save_array = y33
sio.savemat(save_fn, {'y33': save_array})

y44 = np.array(y4)
y44 = y44.astype(np.float64)
save_fn = 'y44.mat'
save_array = y44
sio.savemat(save_fn, {'y44': save_array})


