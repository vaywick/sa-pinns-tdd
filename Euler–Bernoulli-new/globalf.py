import tensorflow as tf
import numpy as np

n_f = 10000 #[20000]
lrate = 0.005
tryerr = 2*10**(-3)
u0_t = 0
v0_t = 0

Nm = 1
Nx, Nt_all= 151, 101
if(Nt_all % Nm !=0 ):
    Nt = int(Nt_all/Nm) + 1
else:
    Nt = int(Nt_all/Nm)
tint = Nt -1

N0=150
Nb=100

tf_iter1= [5000]
newton_iter1= [70000]

num_layer=4
width=48 # 80
layer_sizes=[2]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(1)

doubpa=0
if(doubpa ==1):
    tfdoubstr = tf.float64
else:
    tfdoubstr = tf.float32
    
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
