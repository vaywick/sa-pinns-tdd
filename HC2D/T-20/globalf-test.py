import torch

n_f = 20000
lrate = 0.005
nadap = 10

N0 = 6000
Nb = 3000
u0_t = 0

Nm=4
Nx, Ny, Nt_all= 103, 105, 201
N_x, N_y, Nt_all= 103, 105, 201
if(Nm==1):
    Nt = int(Nt_all/Nm)
else:
    Nt = int(Nt_all/Nm) + 1

N_t= Nt
tint = Nt -1

#X= 1
#Y= 1
#T= 40
    
tf_iter1= [2000, 1500, 1500, 1500]
newton_iter1=[ 500, 500, 500, 500]

num_layer=5
width=64
layer_sizes=[3]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(1)

#doubpa=0
#if(doubpa ==1):
#    tfdoubstr = tf.float64
#else:
#    tfdoubstr = tf.float32

# device: GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

