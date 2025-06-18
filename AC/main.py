import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import math
import matplotlib.gridspec as gridspec
import pickle
import os
#from plotting import newfig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from scipy.interpolate import griddata
#from eager_lbfgs import lbfgs, Struct
from pyDOE import lhs
from SA_PINN import SA_PINN
from globalf import *

tf.random.set_seed(1234)
np.random.seed(1234)

path = './'
file_name = ['Data_flie']

for i in range(len(file_name)):
    file = file_name[i]
    isExists = os.path.exists(path+str(file))
    if not isExists:
        os.makedirs(path+str(file))
        print("%s is not Exists"%file)
    else:
        print("%s is Exists"%file)
        continue

@tf.function
def f_model(model,x,t):
    u = model.model(tf.concat([x,t], 1))
    u_x=tf.gradients(u,x)[0]
    u_xx=tf.gradients(u_x,x)[0]
    u_t=tf.gradients(u,t)[0]
    c1 = tf.constant(.0001, dtype = tfdoubstr)
    c2 = tf.constant(5.0, dtype = tfdoubstr)    
    f = u_t - c1*u_xx + c2*u*u*u - c2*u
    return f


@tf.function
def loss(model,x_f_batch, t_f_batch,
             x0, t0, u0,u_lb,u_ub, x_lb,
         t_lb, x_ub, t_ub,col_weights,u_weights):
    f_u_pred = model.f_model(model,x_f_batch, t_f_batch)
    u0_pred = model.model(tf.concat([x0, t0],1))
    u_lb_pred, u_x_lb_pred = model.u_x_model(x_lb, t_lb)
    u_ub_pred, u_x_ub_pred = model.u_x_model(x_ub, t_ub)
    f0_u_pred=model.f_model(model,x0,t0)

    mse_0_u = tf.reduce_mean(tf.square(u_weights*(u0 - u0_pred)))

    mse_b_u = tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred, u_ub_pred))) + \
              tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred, u_x_ub_pred)))

    mse_f_u = tf.reduce_mean(tf.square(col_weights*f_u_pred))

    return  mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_f_u

for id_t in range(Nm):
   model=SA_PINN("AC.mat",layer_sizes,tf_iter1[id_t],newton_iter1[id_t],f_model=f_model,Loss=loss,N_f= n_f, id_t= id_t, u0_t= u0_t)
   
   model.fit()
#   model.model.save_weights(model.checkPointPath+"/final-%d"%id_t)
   
   Exact_u=model.Exact
   print('exact_u.shape= ', Exact_u.shape)
   X, T = np.meshgrid(model.x,model.t)
   
   X_star = model.X_star
   u_star = model.u_star
   print('u_star.shape= ', u_star.shape)
   
   x0=model.x0
   tb=model.t
   x=model.x
   t=model.t
   
   #lb = np.array([-5.0, -5.0])
   #ub = np.array([5.0, 5])
   lb = X_star.min(0)
   ub = X_star.max(0)
   # Get preds
   u_pred, f_u_pred = model.predict()
   
   print('u_pred.shape = ', u_pred.shape)

   U3_pred = u_pred.reshape((Nt, Nx)).T
   f_U3_pred = f_u_pred.reshape((Nt, Nx)).T

   u0_t= U3_pred[:,-1]

   #find L2 error
   u_star = u_star.T
   u_star = u_star.flatten()[:,None]
   error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
   print('Error u : %e' % (error_u))
   
   ferru = np.mean(np.absolute(f_u_pred))
   print('ferru = ', ferru)
   
   utmp = U3_pred
   f_utmp = f_U3_pred

   if(id_t>0):
      comb_u_pred = np.concatenate((comb_u_pred[:,:-1], utmp), axis=1)
      comb_f_u_pred = np.concatenate((comb_f_u_pred[:,:-1], f_utmp), axis=1)

      Exact_u_i = Exact_u[:, :(tint*(id_t+1) +1)]
      perror_u  = np.linalg.norm((Exact_u_i - comb_u_pred).flatten(),2)
      perror_uEx = np.linalg.norm(Exact_u_i.flatten(),2)

      error_u = perror_u/perror_uEx
      print('Error u : %e' % (error_u))

   else:
      comb_u_pred = U3_pred
      comb_f_u_pred = f_U3_pred

      Exact_u_i = Exact_u[:, :(tint*(id_t+1) +1)]
      perror_u  = np.linalg.norm((Exact_u_i - comb_u_pred).flatten(),2)
      perror_uEx = np.linalg.norm(Exact_u_i.flatten(),2)

      error_u = perror_u/perror_uEx
      print('Error u : %e' % (error_u))

#   if (ferru < tryerr and ferrv < tryerr):
#   	break
#   else:
#   	pass

   print('u_pred.shape after= ', comb_u_pred.shape)


pickle_file1 = open('Data_flie/comb_u_pred.pkl', 'wb')
pickle.dump(comb_u_pred, pickle_file1)
pickle_file1.close()
pickle_file2 = open('Data_flie/comb_f_u_pred.pkl', 'wb')
pickle.dump(comb_f_u_pred, pickle_file2)
pickle_file2.close()

