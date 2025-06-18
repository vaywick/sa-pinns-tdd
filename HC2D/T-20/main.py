import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import time
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import pickle
import os
from globalf import *
from SA_PINN_2D import SA_PINN_2D

path = './'
file_name = ['Data_flie', 'figures']

for i in range(len(file_name)):
    file = file_name[i]
    isExists = os.path.exists(path + str(file))
    if not isExists:
        os.makedirs(path + str(file))
        print("%s is not Exists" % file)
    else:
        print("%s is Exists" % file)
        continue

#torch.manual_seed(1234)
#np.random.seed(1234)

def ensure_grad(tensors):
    for tensor in tensors:
        if not tensor.requires_grad:
            tensor.requires_grad = True

def f_model(sapinn:SA_PINN_2D, x, y, t):
    x_diag = 0.5
    y_diag = 0.5

    ff = 2.*(torch.sin(10*torch.pi*x)*torch.sin(10*torch.pi*y) + torch.exp(-20 * ((x - x_diag) ** 2 + (y - y_diag) ** 2)))

    u = sapinn.u_x_model(x, y, t)

    # Calculate gradients
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    # Calculate the function f
    f = u_t - 0.01 * (u_xx + u_yy) - ff

    return f

def loss(sapinn:SA_PINN_2D, x_f, y_f, t_f, t0, u0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, XY, XT, YT, col_weights, u_weights, ub_weights):
    x_f, y_f, t_f, t0, u0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, XY, XT, YT = \
        [tensor.float().to(device) for tensor in [x_f, y_f, t_f, t0, u0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, XY, XT, YT]]
    
    f_u_pred = f_model(sapinn, x_f, y_f, t_f)

    u0_pred = sapinn.u_x_model(XY[:, 0:1], XY[:, 1:2], t0)
    u_x_lb_pred = sapinn.u_x_model(x_lb, YT[:, 0:1], YT[:, 1:2])
    u_x_ub_pred = sapinn.u_x_model(x_ub, YT[:, 0:1], YT[:, 1:2])
    u_y_lb_pred = sapinn.u_x_model(XT[:, 0:1], y_lb, XT[:, 1:2])
    u_y_ub_pred = sapinn.u_x_model(XT[:, 0:1], y_ub, XT[:, 1:2])

    mse_0_u = torch.mean((u_weights * (u0 - u0_pred)) ** 2)
    mse_b_u = torch.mean((ub_weights * (u_x_lb_pred - u_x_lb)) ** 2) + \
              torch.mean((ub_weights * (u_x_ub_pred - u_x_ub)) ** 2) + \
              torch.mean((ub_weights * (u_y_ub_pred - u_y_ub)) ** 2) + \
              torch.mean((ub_weights * (u_y_lb_pred - u_y_lb)) ** 2)
    mse_f_u = torch.mean((col_weights * f_u_pred) ** 2)

    return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_b_u, mse_f_u

for id_t in range(Nm):
    # Initialize and fit the model
    model = SA_PINN_2D("hc-t20.mat", layer_sizes, tf_iter1[id_t], newton_iter1[id_t], f_model=f_model, Loss=loss, N_f=n_f, id_t=id_t, u0_t= u0_t)
    model.fit()
    model.fit_lbfgs()

    # Save the model
    torch.save(model.model.state_dict(), model.checkPointPath + "/final-%d.pth"%id_t)

    Exact_u = model.u_star_all
#    X, Y, T = np.meshgrid(model.x, model.y, model.t)
    X_star = model.X_star
    u_star = model.u_star

    Ntinter = 5
    
    u_pred = np.zeros((N_y, N_x, Ntinter))
    f_u_pred = np.zeros((N_y, N_x, Ntinter))
    
    dN_t = round(N_t/Ntinter)
    
    for i in range(dN_t+1):
       if (i == dN_t):
         t= model.t[-1]
       else:
         t = model.t[Ntinter*i:Ntinter*(i+1)]
    
       tmp_u_pred, tmp_f_u_pred = model.predict(model.x, model.y, t)
       if(i==0):
          tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, Ntinter)
          tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, Ntinter)
          u_pred = tmp_u_pred
          f_u_pred= tmp_f_u_pred
    
       elif(i!=dN_t):
          tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, Ntinter)
          tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, Ntinter)
          u_pred = np.concatenate((u_pred, tmp_u_pred), axis=2)
          f_u_pred = np.concatenate((f_u_pred, tmp_f_u_pred), axis=2)
    
       else:
          tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, 1)
          tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, 1)
          u_pred = np.concatenate((u_pred, tmp_u_pred), axis=2)
          f_u_pred = np.concatenate((f_u_pred, tmp_f_u_pred), axis=2)

    Exact_u = Exact_u.reshape(Ny, Nx, Nt_all)

    U3_pred = u_pred.reshape((Ny, Nx, Nt))
    f_U3_pred = f_u_pred.reshape((Ny, Nx, Nt))

    u0_t= U3_pred[:,:,-1]

    u_pred_tmp= u_pred.flatten()[:,None]
    error_u = np.linalg.norm(u_star-u_pred_tmp,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))

#    ferr = np.linalg.norm(f_u_pred, 2)
    ferru = np.mean(np.absolute(f_u_pred))
    print('ferru = ', ferru)

    utmp = U3_pred
    f_utmp = f_U3_pred


    if(id_t>0):
       comb_u_pred = np.concatenate((comb_u_pred[:,:,:-1], utmp), axis=2)
       comb_f_u_pred = np.concatenate((comb_f_u_pred[:,:,:-1], f_utmp), axis=2)

       Exact_u_i = Exact_u[:,:, :(tint*(id_t+1) +1)]
       perror_u  = np.linalg.norm((Exact_u_i - comb_u_pred).flatten(),2)
       perror_uEx = np.linalg.norm(Exact_u_i.flatten(),2)

       error_u = perror_u/perror_uEx
       print('Error u: %e' % (error_u))

    else:
       comb_u_pred = U3_pred
       comb_f_u_pred = f_U3_pred

       Exact_u_i = Exact_u[:,:, :(tint*(id_t+1) +1)]
       perror_u  = np.linalg.norm((Exact_u_i - comb_u_pred).flatten(),2)
       perror_uEx = np.linalg.norm(Exact_u_i.flatten(),2)

       error_u = perror_u/perror_uEx
       print('Error u: %e' % (error_u))

#    if (ferru < tryerr and ferrv < tryerr):
#        break
#    else:
#        pass

    print('u_pred.shape after= ', comb_u_pred.shape)

pickle_file1 = open('Data_flie/comb_u_pred.pkl', 'wb')
pickle.dump(comb_u_pred, pickle_file1)
pickle_file1.close()
pickle_file2 = open('Data_flie/comb_f_u_pred.pkl', 'wb')
pickle.dump(comb_f_u_pred, pickle_file2)
pickle_file2.close()

