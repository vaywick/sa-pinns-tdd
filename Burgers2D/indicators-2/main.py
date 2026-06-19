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

startt= time.time()

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

def uv_model(model, x, y, t):
    x = x.requires_grad_(True).float().to(device)
    y = y.requires_grad_(True).float().to(device)
    t = t.requires_grad_(True).float().to(device)

    uv = model(torch.cat([x, y, t], dim=1))
    u = uv[:,0:1]
    v = uv[:,1:2]
    return u, v

def ensure_grad(tensors):
    for tensor in tensors:
        if not tensor.requires_grad:
            tensor.requires_grad = True

def f_model(model, x, y, t):
    u, v = uv_model(model, x, y, t)
    u_t = torch.autograd.grad(u, inputs=t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, inputs=y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, inputs=x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, inputs=y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    v_t = torch.autograd.grad(v, inputs=t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, inputs=x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, inputs=y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, inputs=x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, inputs=y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    nu = 0.01/np.pi
    f_u=u_t - nu*(u_xx+u_yy) +u*u_x+v*u_y
    f_v=v_t - nu*(v_xx+v_yy) +u*v_x+v*v_y

    return f_u,f_v

def loss(model, x_f, y_f, t_f, t0, u0, v0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, v_x_lb, v_x_ub, v_y_lb, v_y_ub, XY, XT, YT, col_weights, u_weights, ub_weights, col_v_weights, v_weights, vb_weights):
    x_f, y_f, t_f, t0, u0, v0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, v_x_lb, v_x_ub, v_y_lb, v_y_ub, XY, XT, YT = \
        [tensor.float().to(device) for tensor in [x_f, y_f, t_f, t0, u0, v0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, v_x_lb, v_x_ub, v_y_lb, v_y_ub, XY, XT, YT]]
    
    f_u_pred, f_v_pred = f_model(model, x_f, y_f, t_f)

    u0_pred, v0_pred = uv_model(model, XY[:, 0:1], XY[:, 1:2], t0)
    u_x_lb_pred, v_x_lb_pred = uv_model(model, x_lb, YT[:, 0:1], YT[:, 1:2])
    u_x_ub_pred, v_x_ub_pred = uv_model(model, x_ub, YT[:, 0:1], YT[:, 1:2])
    u_y_lb_pred, v_y_lb_pred = uv_model(model, XT[:, 0:1], y_lb, XT[:, 1:2])
    u_y_ub_pred, v_y_ub_pred = uv_model(model, XT[:, 0:1], y_ub, XT[:, 1:2])

    mse_0_u = torch.mean((u_weights.float().to(device) * (u0.float().to(device) - u0_pred)) ** 2)
    mse_0_v = torch.mean((v_weights.float().to(device) * (v0.float().to(device) - v0_pred)) ** 2)

    mse_b_u = torch.mean(((ub_weights*(u_x_lb_pred - u_x_lb)) ** 2)) + \
              torch.mean(((ub_weights*(u_x_ub_pred - u_x_ub)) ** 2)) + \
              torch.mean(((ub_weights*(u_y_ub_pred - u_y_ub)) ** 2)) + \
              torch.mean(((ub_weights*(u_y_lb_pred - u_y_lb)) ** 2)) + \
              torch.mean(((vb_weights*(v_x_lb_pred - v_x_lb)) ** 2)) + \
              torch.mean(((vb_weights*(v_x_ub_pred - v_x_ub)) ** 2)) + \
              torch.mean(((vb_weights*(v_y_ub_pred - v_y_ub)) ** 2)) + \
              torch.mean(((vb_weights*(v_y_lb_pred - v_y_lb)) ** 2))

    mse_f_u = torch.mean((col_weights.float().to(device) * f_u_pred) ** 2)
    mse_f_v = torch.mean((col_v_weights.float().to(device) * f_v_pred) ** 2)

    return mse_0_u + mse_b_u + mse_f_u + mse_0_v + mse_f_v , mse_0_u + mse_0_u, mse_b_u + mse_b_u, mse_f_u + mse_f_v

for id_t in range(Nm):
    # Initialize and fit the model
    model = SA_PINN_2D("burgers2d.mat", layer_sizes, tf_iter1[id_t], newton_iter1[id_t], f_model=f_model, Loss=loss, N_f=n_f, id_t=id_t, u0_t= u0_t, v0_t= v0_t)
    model.fit()
    model.fit_lbfgs()

    # Save the model
    torch.save(model.u_model.state_dict(), model.checkPointPath + "/final.pth")

    Exact_u = model.u_star_all
    Exact_v = model.v_star_all

    X_star = model.X_star
    u_star = model.u_star#.flatten()[:,None]
    v_star = model.v_star#.flatten()[:,None]

    Ntinter = 5
    
    u_pred = np.zeros((N_y, N_x, Ntinter))
    f_u_pred = np.zeros((N_y, N_x, Ntinter))
    v_pred = np.zeros((N_y, N_x, Ntinter))
    f_v_pred = np.zeros((N_y, N_x, Ntinter))
    
    dN_t = round(N_t/Ntinter)
    
    for i in range(dN_t+1):
       if (i == dN_t):
         t= model.t[-1]
       else:
         t = model.t[Ntinter*i:Ntinter*(i+1)]
    
       tmp_u_pred, tmp_f_u_pred, tmp_v_pred, tmp_f_v_pred = model.predict(model.x, model.y, t)
       if(i==0):
          tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, Ntinter)
          tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, Ntinter)
          u_pred = tmp_u_pred
          f_u_pred= tmp_f_u_pred
    
          tmp_v_pred = tmp_v_pred.reshape(N_y, N_x, Ntinter)
          tmp_f_v_pred = tmp_f_v_pred.reshape(N_y, N_x, Ntinter)
          v_pred = tmp_v_pred
          f_v_pred= tmp_f_v_pred
    
       elif(i!=dN_t):
          tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, Ntinter)
          tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, Ntinter)
          u_pred = np.concatenate((u_pred, tmp_u_pred), axis=2)
          f_u_pred = np.concatenate((f_u_pred, tmp_f_u_pred), axis=2)
    
          tmp_v_pred = tmp_v_pred.reshape(N_y, N_x, Ntinter)
          tmp_f_v_pred = tmp_f_v_pred.reshape(N_y, N_x, Ntinter)
          v_pred = np.concatenate((v_pred, tmp_v_pred), axis=2)
          f_v_pred = np.concatenate((f_v_pred, tmp_f_v_pred), axis=2)
    
       else:
          tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, 1)
          tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, 1)
          u_pred = np.concatenate((u_pred, tmp_u_pred), axis=2)
          f_u_pred = np.concatenate((f_u_pred, tmp_f_u_pred), axis=2)

          tmp_v_pred = tmp_v_pred.reshape(N_y, N_x, 1)
          tmp_f_v_pred = tmp_f_v_pred.reshape(N_y, N_x, 1)
          v_pred = np.concatenate((v_pred, tmp_v_pred), axis=2)
          f_v_pred = np.concatenate((f_v_pred, tmp_f_v_pred), axis=2)

    Exact_u = Exact_u.reshape(Ny, Nx, Nt_all)
    Exact_v = Exact_v.reshape(Ny, Nx, Nt_all)

    U3_pred = u_pred.reshape((Ny, Nx, Nt))
    V3_pred = v_pred.reshape((Ny, Nx, Nt))
    f_U3_pred = f_u_pred.reshape((Ny, Nx, Nt))
    f_V3_pred = f_v_pred.reshape((Ny, Nx, Nt))

    u0_t= U3_pred[:,:,-1]
    v0_t= V3_pred[:,:,-1]

    u_pred_tmp= u_pred.flatten()[:,None]
    v_pred_tmp= v_pred.flatten()[:,None]
    error_u = np.linalg.norm(u_star-u_pred_tmp,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred_tmp,2)/np.linalg.norm(v_star,2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))

    ferru = np.mean(np.absolute(f_u_pred))
    ferrv = np.mean(np.absolute(f_v_pred))
    print('ferru = ', ferru)
    print('ferrv = ', ferrv)

    utmp = U3_pred
    vtmp = V3_pred
    f_utmp = f_U3_pred
    f_vtmp = f_V3_pred


    if(id_t>0):
       comb_u_pred = np.concatenate((comb_u_pred[:,:,:-1], utmp), axis=2)
       comb_v_pred = np.concatenate((comb_v_pred[:,:,:-1], vtmp), axis=2)
       comb_f_u_pred = np.concatenate((comb_f_u_pred[:,:,:-1], f_utmp), axis=2)
       comb_f_v_pred = np.concatenate((comb_f_v_pred[:,:,:-1], f_vtmp), axis=2)

       Exact_u_i = Exact_u[:,:, :(tint*(id_t+1) +1)]
       Exact_v_i = Exact_v[:,:, :(tint*(id_t+1) +1)]
       perror_u  = np.linalg.norm((Exact_u_i - comb_u_pred).flatten(),2)
       perror_v  = np.linalg.norm((Exact_v_i - comb_v_pred).flatten(),2)
       perror_uEx = np.linalg.norm(Exact_u_i.flatten(),2)
       perror_vEx = np.linalg.norm(Exact_v_i.flatten(),2)

       error_u = perror_u/perror_uEx
       error_v = perror_v/perror_vEx
       print('Error u: %e' % (error_u))
       print('Error v: %e' % (error_v))

    else:
       comb_u_pred = U3_pred
       comb_v_pred = V3_pred
       comb_f_u_pred = f_U3_pred
       comb_f_v_pred = f_V3_pred

       Exact_u_i = Exact_u[:,:, :(tint*(id_t+1) +1)]
       Exact_v_i = Exact_v[:,:, :(tint*(id_t+1) +1)]
       perror_u  = np.linalg.norm((Exact_u_i - comb_u_pred).flatten(),2)
       perror_v  = np.linalg.norm((Exact_v_i - comb_v_pred).flatten(),2)
       perror_uEx = np.linalg.norm(Exact_u_i.flatten(),2)
       perror_vEx = np.linalg.norm(Exact_v_i.flatten(),2)

       error_u = perror_u/perror_uEx
       error_v = perror_v/perror_vEx
       print('Error u: %e' % (error_u))
       print('Error v: %e' % (error_v))

#    if (ferru < tryerr and ferrv < tryerr):
#        break
#    else:
#        pass

pickle_file1 = open('Data_flie/comb_u_pred.pkl', 'wb')
pickle.dump(comb_u_pred, pickle_file1)
pickle_file1.close()
pickle_file2 = open('Data_flie/comb_f_u_pred.pkl', 'wb')
pickle.dump(comb_f_u_pred, pickle_file2)
pickle_file2.close()
pickle_file1 = open('Data_flie/comb_v_pred.pkl', 'wb')
pickle.dump(comb_v_pred, pickle_file1)
pickle_file1.close()
pickle_file2 = open('Data_flie/comb_f_v_pred.pkl', 'wb')
pickle.dump(comb_f_v_pred, pickle_file2)
pickle_file2.close()

endt = time.time()

computime= endt - startt

print('time consuming : ', computime)

