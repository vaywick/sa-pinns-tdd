import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import math
import matplotlib.gridspec as gridspec
import pickle
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from scipy.interpolate import griddata
from pyDOE import lhs
from SA_PINN import SA_PINN
from globalf import *

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

#tf.random.set_seed(1234)
#np.random.seed(1234)

@tf.function
def f_model(model,x,t):
    u, u_t, u_xx = model.uv_model(x,t)
        
    u_tt=tf.gradients(u_t,t)[0]
    u_xxx = tf.gradients(u_xx, x)[0]
    u_xxxx = tf.gradients(u_xxx, x)[0]

    f_u= u_tt + u_xxxx + u - (2-np.pi**2)*tf.sin(x)*tf.cos(np.pi*t)
        
    return f_u

@tf.function
def loss(model,x_f_batch, t_f_batch,
             x0, t0, u0, u0td,u_lb_xx,u_ub_xx,u_lb,u_ub, x_lb,
             t_lb, x_ub, t_ub,col_weights,u_weights,ut_weights,u_lb_weights,u_ub_weights, u_lb_xx_weights, u_ub_xx_weights):
    u0_pred,u0td_pred,_=model.uv_model(x0,t0)
    u_lb_pred,_,u_lb_xx_pred=model.uv_model(x_lb,t_lb)
    u_ub_pred,_,u_ub_xx_pred=model.uv_model(x_ub,t_ub)
    f_u_pred=model.f_model(model,x_f_batch,t_f_batch)

    mse_0_u = tf.reduce_mean(tf.square(u_weights*(u0 - u0_pred)))+tf.reduce_mean(tf.square(ut_weights*(u0td - u0td_pred)))

    mes_b_u=tf.reduce_mean(tf.square(u_lb_weights*(u_lb_pred - u_lb)))+tf.reduce_mean(tf.square(u_ub_weights*(u_ub_pred - u_ub))) + \
            tf.reduce_mean(tf.square(u_lb_xx_weights*(u_lb_xx_pred - u_lb_xx)))+tf.reduce_mean(tf.square(u_ub_xx_weights*(u_ub_xx_pred - u_ub_xx)))
    mes_f_u=tf.reduce_mean(tf.square(col_weights*f_u_pred))

    return mse_0_u+mes_b_u+mes_f_u,mse_0_u,mes_f_u

for id_t in range(Nm):
   model=SA_PINN("data.mat",layer_sizes,tf_iter1[id_t],newton_iter1[id_t],f_model=f_model,Loss=loss,lbfgs_lr=0.8, N_f= n_f, id_t= id_t, u0_t= u0_t, v0_t= v0_t)
   
   model.fit()
   model.model.save_weights(model.checkPointPath+"/final")
   
   Exact_u=model.Exact_u
   X, T = np.meshgrid(model.x,model.t)
   
   X_star = model.X_star
   u_star = model.u_star
   
   x0=model.x0
   tb=model.t
   x=model.x
   t=model.t
   
   lb = X_star.min(0)
   ub = X_star.max(0)
   print('u_star.shape= ', u_star.shape)
   
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
   print('Error u: %e' % (error_u))
   
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

