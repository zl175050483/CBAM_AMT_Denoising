# prediction
import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from g12 import SCNet18

###data and model loading###
noise_data = 'noise.mat'
noise_data = sio.loadmat(noise_data)
noise_data = noise_data['noise'] 

data_in_channel=4
data_out_channel=4
data_size=256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SCNet18().to(device)
weights_path = "./net.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))

###preprocessing###
L = noise_data.shape[1]
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

###identification###
model.eval()
with torch.no_grad():
    output = model(p_data.to(device))
    _, predicted = torch.max(output.data, 1)
predicted = predicted.cpu().detach().numpy()
output1 = np.zeros([Ln,data_out_channel,data_size])
for i in range(Ln):
    output1[i,:,:]=predicted[i]

denoising4 = np.zeros([data_out_channel,L])
for k in range(data_out_channel):
    denoising1 = np.zeros([L,L])
    for i in range(Ln):
        denoising1[i,i:i+data_size]=output1[i,k,:]

    denoising2 = np.sum(denoising1,axis=0)
    denoising3 = denoising2[0:L]
    for i in range(data_size-1):
        denoising3[i]=denoising2[i]/(i+1)
        denoising3[Ln+i]=denoising2[Ln+i]/(data_size-i)
    denoising3[data_size-1:Ln]=denoising2[data_size-1:Ln]/data_size
    denoising4[k,:]=denoising3
    o=np.where(denoising4>0)
    denoising4[o]=1
    o=np.array(o)
    o=o[1]

# save output
save_fn = 'predicted.mat'
save_array = predicted
sio.savemat(save_fn, {'predicted': save_array})

save_fn = 'o.mat'
save_array = o
sio.savemat(save_fn, {'o': save_array})

#plot
x = range(0, L)
y1 = noise_data[0,:]
plt.plot(x, y1, 'b.-')
plt.plot(o, y1[o], 'r.-')
plt.xlabel('Time')
plt.ylabel('Ampulitude')
plt.show()