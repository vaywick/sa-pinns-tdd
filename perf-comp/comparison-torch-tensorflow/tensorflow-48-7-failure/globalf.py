import tensorflow as tf

n_f = 20000
N0 = 3000#7000, 5000
Nb = 300#3000, 1000
lrate = 0.005
ainit = 0.1
a_trainable=False
nadap = 10

tf_iter1=200
newton_iter1=200

num_layer=7
width=48
layer_sizes=[3]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(1)

doubpa=0
if(doubpa ==1):
    doubstr = 'float64'
    tfdoubstr = tf.float64
else:
    doubstr = 'float32'
    tfdoubstr = tf.float32

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#tf.keras.backend.set_floatx(doubstr)
