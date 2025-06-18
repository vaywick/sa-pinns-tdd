import numpy as np
import torch

n_f = 20000 #[20000]
lrate = 0.005
tryerr = 2*10**(-3)
u0_t = 0
v0_t = 0

Nm = 7
Nx, Nt_all= 385, 141
if(Nt_all % Nm !=0):
    Nt = int(Nt_all/Nm) + 1
else:
    Nt = int(Nt_all/Nm)# + 1
tint = Nt -1

#N0 = 256
N0 = 350
Nb = 20

tf_iter1= [12000, 12000, 12000, 12000, 10000, 8000, 6000]#, 10000, 10000]
newton_iter1= [25000, 20000, 15000, 12000, 10000, 8000, 6000]#, 30000, 30000]
max_retry=1
tryerr = 5*10**(2)

num_layer=4
width=80
layer_sizes=[2]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(2)

# 训练设备为GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

