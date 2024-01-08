#prediction
import torch
import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
from g15 import SCNet18

###data and model loading###
origin = 'ey1.mat'
origin = sio.loadmat(origin)
origin = origin['ey1'] 
noise_data1 = 'noise.mat'
noise_data1 = sio.loadmat(noise_data1)
noise_data1 = noise_data1['noise'] 

data_in_channel=4
data_out_channel=4
data_size=256
noise_data = noise_data1 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SCNet18().to(device)
weights_path = "./net.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))

###preprocessing###
L = origin.shape[1]
Ln = L-data_size+1
noise_data_s=np.zeros([Ln,data_in_channel,data_size])
for k in range(data_in_channel):
    for i in range(Ln):
        noise_data_s[i,k,:] = noise_data[k,i:i+data_size]
#noise_data_s = noise_data_s[:,np.newaxis,:]

def normalization(data,_range):
    return data / _range
    
range_p = np.max(abs(noise_data_s))
p_norm = normalization(noise_data_s,range_p)
np.save('range_p',range_p)

p_data=torch.from_numpy(p_norm)  
p_data = p_data.type(torch.FloatTensor)

###denoising###
model.eval()
with torch.no_grad():
    output = model(p_data.to(device))
    #output = torch.squeeze(output).cpu().detach().numpy()
    output = output.cpu().detach().numpy()

output = output*range_p
#output = output[0:Ln,:]

denoising4 = np.zeros([data_out_channel,L])
for k in range(data_out_channel):
    denoising1 = np.zeros([L,L])
    for i in range(Ln):
        denoising1[i,i:i+data_size]=output[i,k,:]

    denoising2 = np.sum(denoising1,axis=0)
    denoising3 = denoising2[0:L]
    for i in range(data_size-1):
        denoising3[i]=denoising2[i]/(i+1)
        denoising3[Ln+i]=denoising2[Ln+i]/(data_size-i)
    denoising3[data_size-1:Ln]=denoising2[data_size-1:Ln]/data_size
    denoising4[k,:]=denoising3
denoising5 = np.sum(denoising4,axis=0)/2

#plot
k=3 #i-th component's denoising result
x = range(0, L)
y1 = origin[k].flatten()
y2 = denoising4[k].flatten()
plt.plot(x, y1, 'b.-')
plt.plot(x, y2, 'r.-')
plt.xlabel('Time')
plt.ylabel('Ampulitude')
plt.show()

###SNR###
#before denoising
origSignal = origin[k]
errorSignal = origin[k]-noise_data1[k]
signal_2 = sum(origSignal.flatten()**2)
noise_2 = sum(errorSignal.flatten()**2)
SNRValues1 = 10*math.log10(signal_2/noise_2)
print(SNRValues1)

#after denoising
origSignal = origin[k]
errorSignal = origin[k]-denoising4[k]
signal_2 = sum(origSignal.flatten()**2)
noise_2 = sum(errorSignal.flatten()**2)
SNRValues2 = 10*math.log10(signal_2/noise_2)
print(SNRValues2)  

save_fn = 'denoising4.mat'
save_array = denoising4
sio.savemat(save_fn, {'denoising4': save_array})