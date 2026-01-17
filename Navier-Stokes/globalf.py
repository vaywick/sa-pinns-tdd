import torch

n_f = 50000
lrate = 0.005
nadap = 10

N0 = 10000
Nb = 2000
u0_t = 0
v0_t = 0
p0_t = 0

Nm = 4
Nx, Ny, Nt_all= 153, 155, 61
N_x, N_y, Nt_all= 153, 155, 61
#Nt = int(Nt_all/Nm) + 1
if(Nm==1):
    Nt = int(Nt_all/Nm)
else:
    Nt = int(Nt_all/Nm) + 1

N_t= Nt
tint = Nt -1

#X= 10
#Y= 10
#T= 2*pi
    
tf_iter1=[30000, 30000, 30000, 30000]
newton_iter1=[70000, 70000, 70000, 70000]

num_layer=6
width=96
layer_sizes=[3]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(3)

#doubpa=0
#if(doubpa ==1):
#    tfdoubstr = tf.float64
#else:
#    tfdoubstr = tf.float32

# device: GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

