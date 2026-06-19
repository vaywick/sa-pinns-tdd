import torch

n_f = 20000
lrate = 0.005
nadap = 10

N0 = 4000
Nb = 2000
u0_t = 0
v0_t = 0

Nm = 4
Nx, Ny, Nt_all= 101, 101, 401
N_x, N_y, Nt_all= 101, 101, 401
#Nt = int(Nt_all/Nm) + 1
if(Nm==1):
    Nt = int(Nt_all/Nm)
else:
    Nt = int(Nt_all/Nm) + 1

N_t= Nt
tint = Nt -1

#X= 1
#Y= 1
#T= 10
    
tf_iter1=[22000, 18000, 17000, 17000]
newton_iter1=[40000, 30000, 15000, 15000]

num_layer=6
width=64
layer_sizes=[3]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(2)

#doubpa=0
#if(doubpa ==1):
#    tfdoubstr = tf.float64
#else:
#    tfdoubstr = tf.float32

# device: GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

