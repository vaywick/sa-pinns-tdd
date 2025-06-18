import tensorflow as tf
import numpy as np

n_f = 5000 #[20000]
lrate = 0.005
tryerr = 2*10**(-3)
u0_t = 0

Nm = 2
Nx, Nt_all= 512, 201
Nt = int(Nt_all/Nm) + 1
tint = Nt -1

#N0 = 256
N0 = Nx
Nb = Nt

tf_iter1= [10000, 10000]#, 10000, 10000]
newton_iter1= [50000, 50000]#, 30000, 30000]
max_retry=1
tryerr = 5*10**(2)

num_layer=8
width=20
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

