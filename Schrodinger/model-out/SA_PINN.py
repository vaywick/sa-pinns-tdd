import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import scipy.io
from pyDOE import lhs
import pickle
import os
from globalf import *
import datetime

class SA_PINN:
    def __DefaultLoss(self,x_f_batch, t_f_batch,
             x0, t0, u0,u_lb,u_ub, x_lb,
             t_lb, x_ub, t_ub, col_weights, u_weights):
#             t_lb, x_ub, t_ub,SA_weight):
        u0_pred = self.u_model(torch.cat([x0, t0], 1))
        u_lb_pred, u_x_lb_pred = self.u_x_model(self.model, x_lb, t_lb)
        u_ub_pred, u_x_ub_pred = self.u_x_model(self.model, x_ub, t_ub)
        f_u_pred = self.f_model(self.model, x_f_batch, t_f_batch)
        
        mse_0_u = torch.mean((u_weights * (u0 - u0_pred))**2)
        mse_b_u = torch.mean((u_lb_pred - u_ub_pred)**2) + torch.mean((u_x_lb_pred - u_x_ub_pred)**2)
        mse_f_u = torch.mean((col_weights * f_u_pred)**2)
        
        return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_b_u, mse_f_u
    def u_model(self, x, t):
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        
        u = self.model(torch.cat([x, t], dim=1))
        #u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        return u

    def __init__(self,mat_filename,layers:[],adam_iter:int,newton_iter:int,f_model,Loss=__DefaultLoss,lbfgs_lr=0.8,N_f=10000, id_t=0, u0_t= 0,checkPointPath="./checkPoint"):
        self.N_f=N_f
        self.u0_t= u0_t
        self.id_t= id_t
        self.lables_0=()
        self.lables_lb=()
        self.lables_ub=()
        self.__Loadmat(mat_filename)
        self.layers=layers
        self.sizes_w=[]
        self.sizes_b=[]
#        sizes_w=[]
#        sizes_b=[]
        self.lbfgs_lr=lbfgs_lr
        self.SA_weights={
            'col_weights':nn.Parameter(torch.full((N_f, 1), 100.0, device=device)),
            'u_weights':nn.Parameter(torch.full((self.x0.shape[0], 1), 100.0, device=device)),         
            'u_lb_weights':nn.Parameter(torch.full((self.x_lb.shape[0], 1), 100.0, device=device)),         
            'u_ub_weights':nn.Parameter(torch.full((self.x_ub.shape[0], 1), 100.0, device=device)),         
                         }

        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

        # Define neural network architecture
        class NeuralNet(nn.Module):
            def __init__(self, layer_sizes):
                super(NeuralNet, self).__init__()
                layers = []
                input_size = layer_sizes[0]
                for output_size in layer_sizes[1:-1]:
                    layers.append(nn.Linear(input_size, output_size))
                    layers.append(nn.Tanh())
                    input_size = output_size
                layers.append(nn.Linear(input_size, layer_sizes[-1]))
                self.network = nn.Sequential(*layers)
#                network = nn.Sequential(*layers)
            
            def forward(self, x):
#                return network(x)
                return self.network(x)

            def load(self, path):
                self.network.load_state_dict(torch.load(path))
                return self.network
#                self.model.eval()


        self.checkPointPath = f"{checkPointPath}/model-%d"%(id_t)
        if not os.path.exists(self.checkPointPath):
            os.makedirs(self.checkPointPath)

        if(self.id_t==0):
           self.model = NeuralNet(self.layers)
        else:
           self.model = NeuralNet(self.layers)
           path_to_model = "checkPoint/model-%d/final.pth"%(id_t-1)
           self.model.load_state_dict(torch.load(path_to_model))

        print(self.model)

        self.model = self.model.cuda()

        self.Loss=Loss
        self.adam_iter=adam_iter
        self.newton_iter=newton_iter
        self.f_model=f_model
#        self.ux_model=ux_model


    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def ggrad(self, model, x_f_batch, t_f_batch, x0_batch, t0_batch, lables_0, lables_lb, lables_ub, x_lb, t_lb, x_ub, t_ub):
        x_f_batch = x_f_batch.to(device).requires_grad_(True)
        t_f_batch = t_f_batch.to(device).requires_grad_(True)
        x0_batch = x0_batch.to(device).requires_grad_(True)
        t0_batch = t0_batch.to(device).requires_grad_(True)
        lables_0 = tuple([lable.to(device).requires_grad_(True) for lable in lables_0])#lable_0.to(device).requires_grad_(True)
        x_lb = x_lb.to(device).requires_grad_(True)
        t_lb = t_lb.to(device).requires_grad_(True)
        x_ub = x_ub.to(device).requires_grad_(True)
        t_ub = t_ub.to(device).requires_grad_(True)
#        col_weights = self.col_weights.to(device).requires_grad_(True)
#        u_weights = self.u_weights.to(device).requires_grad_(True)
#        ub_weights = self.ub_weights.to(device).requires_grad_(True)
    
        model.zero_grad()
    
        # Forward pass
        loss_value, mse_0, mse_b, mse_f = self.Loss(self, x_f_batch, t_f_batch, x0_batch, t0_batch, lables_0, lables_lb, lables_ub, x_lb, t_lb, x_ub, t_ub, self.SA_weights)
    
        # Backward pass
        loss_value.backward(retain_graph=True)
    
        grads = [param.grad.clone() for param in model.parameters()]
    
        model.zero_grad()
    
        loss_value.backward(retain_graph=True)
        #grads_col = self.col_weights.grad.clone()
        #grads_u = self.u_weights.grad.clone()
        SA_grads=[SA_weight.grad.clone() for key,SA_weight in self.SA_weights.items()]
    
        return loss_value.item(), mse_0.item(), mse_b.item(), mse_f.item(), grads, SA_grads

    def fit(self):
    
        # Set batch size for collocation points
        batch_sz = self.N_f
        n_batches = self.N_f // batch_sz
    
        start_time = time.time()
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.005, betas=(0.99, 0.999))
        SA_optimizers=[optim.Adam([SA_weight], lr=0.005, betas=(0.99, 0.999)) for key,SA_weight in self.SA_weights.items()]
        print("starting Adam training")
    
        loss_history = []
        resi_indicator = []
        weight_history = []
        ad_n=0
        uerrorad_history = []
        verrorad_history = []
        # For mini-batch (if used)
        for epoch in range(self.adam_iter):
            for i in range(n_batches):
    
                x0_batch = torch.tensor(self.x0, dtype=torch.float32)
                t0_batch = torch.tensor(self.t0, dtype=torch.float32)
                #u0_batch = torch.tensor(self.u0, dtype=torch.float32)
    
                x_f_batch = torch.tensor(self.x_f[i*batch_sz:(i*batch_sz + batch_sz),], dtype=torch.float32)
                t_f_batch = torch.tensor(self.t_f[i*batch_sz:(i*batch_sz + batch_sz),], dtype=torch.float32)
    
                loss_value, mse_0, mse_b, mse_f, grads, SA_grads = self.ggrad(self.model, x_f_batch, t_f_batch, x0_batch, t0_batch, self.lables_0, self.lables_lb, self.lables_ub, self.x_lb, self.t_lb, self.x_ub, self.t_ub)
    
                # Apply gradients to the model
                optimizer.zero_grad()
                for param, grad in zip(self.model.parameters(), grads):
                    param.grad = grad
                optimizer.step()

                index=0
                for key,SA_weights in self.SA_weights.items(): 
                    SA_optimizers[index].zero_grad()
                    SA_weights.grad = -SA_grads[index]
                    SA_optimizers[index].step()
                    index+=1
    
            if (epoch+1) % 5000 == 0:
                torch.save(self.model.state_dict(), f"{self.checkPointPath}/adam{epoch+1}.pth")

#            error_u_value, error_v_value = self.error_u()
#            uerrorad_history.append(error_u_value)
#            verrorad_history.append(error_v_value)

            if (epoch+1) % 100 == 0:
                elapsed = time.time() - start_time
                error_u_value, error_v_value = self.error_u()
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e, Error u,v: %.4e, %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value, error_u_value, error_v_value))
    
                start_time = time.time()
            loss_history.append(loss_value)
   
            if (epoch+1) >= 4000 and (epoch+1) % 500 == 0:
                resi_indicator.append(self.SA_weights['col_v_weights'])
                # pde_indicator.append(mse_f)
                
                current_weights = torch.cat([param.detach().view(-1) for param in self.model.parameters()])
                weight_history.append(current_weights)

            if (epoch+1) >= 4500 and (epoch+1) % 500 == 0:
                rela_err_l2 = torch.norm((resi_indicator[-2] - resi_indicator[-1]), p=2) / torch.norm(resi_indicator[-2], p=2)
                
                weight_err_l2 = torch.norm((weight_history[-2] - weight_history[-1]), p=2) / torch.norm(weight_history[-2], p=2)
                
                if rela_err_l2 < 1e-3 and weight_err_l2 < 1e-2:
                    ad_n += 1
                    if ad_n >= 4:
                        break
                    else:
                        pass

#        with open('Data_flie/lossad_value-%d.pkl'%self.id_t, 'wb') as f:
#            pickle.dump(loss_history, f)
#
#        with open('Data_flie/uerrorad_value-%d.pkl'%self.id_t, 'wb') as f:
#            pickle.dump(uerrorad_history, f)
#
#        with open('Data_flie/verrorad_value-%d.pkl'%self.id_t, 'wb') as f:
#            pickle.dump(verrorad_history, f)

    def fit_lbfgs(self):
    
        batch_sz = self.N_f
        n_batches = self.N_f // batch_sz
    
        start_time = time.time()
        
        optimizer = optim.LBFGS(self.model.parameters(), lr=0.8, tolerance_grad=1e-05, tolerance_change=1e-09)
    
        print("starting L-BFGS training")
    
        loss_indicator = []
        pde_indicator = []
        loss_history_lb = []
        uerror_history_lb = []
        verror_history_lb = []
        for epoch in range(self.newton_iter):
            for i in range(n_batches):
    
                x0_batch = torch.tensor(self.x0, dtype=torch.float32)
                t0_batch = torch.tensor(self.t0, dtype=torch.float32)
                u0_batch = torch.tensor(self.u0, dtype=torch.float32)
    
                x_f_batch = torch.tensor(self.x_f[i*batch_sz:(i*batch_sz + batch_sz),], dtype=torch.float32)
                t_f_batch = torch.tensor(self.t_f[i*batch_sz:(i*batch_sz + batch_sz),], dtype=torch.float32)
    
                loss_value, mse_0, mse_b, mse_f, grads, SA_grads = self.ggrad(self.model, x_f_batch, t_f_batch, x0_batch, t0_batch, self.lables_0, self.lables_lb, self.lables_ub, self.x_lb, self.t_lb, self.x_ub, self.t_ub)
    
                def closure():
                    optimizer.zero_grad()
                    for param, grad in zip(self.model.parameters(), grads):
                        param.grad = grad
    
                    return loss_value       
                
                optimizer.step(closure)
    
            if (epoch+1) % 2000 == 0:
                torch.save(self.model.state_dict(), f"{self.checkPointPath}/lbfgs{epoch+1}.pth")
    
#            error_u_value, error_v_value = self.error_u()
#            uerror_history_lb.append(error_u_value)
#            verror_history_lb.append(error_v_value)
            if (epoch+1) % 100 == 0:
                elapsed = time.time() - start_time
                error_u_value, error_v_value = self.error_u()
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e, Error u,v: %.4e, %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value, error_u_value, error_v_value))
    
                start_time = time.time()

            if (epoch+1)>= 5000 and (epoch+1) % 500 == 0:
               loss_indicator.append(loss_value)
               pde_indicator.append(mse_f)
            if (epoch+1)>= 5500 and (epoch+1) % 500 == 0:
               if abs((loss_indicator[-2]-loss_indicator[-1])/loss_indicator[-2])<8e-2 or abs((pde_indicator[-2]-pde_indicator[-1])/pde_indicator[-2])<8e-2:
                   break
               else:
                   pass

            loss_history_lb.append(loss_value)#.item())

            if (epoch+1) % 2000 == 0:
                with open('Data_flie/lossl_value-%d.pkl'%self.id_t, 'wb') as f:
                    pickle.dump(loss_history_lb, f)
    
#        with open('Data_flie/lossl_value-%d.pkl'%self.id_t, 'wb') as f:
#            pickle.dump(loss_history_lb, f)
#
#        with open('Data_flie/uerrorl_value-%d.pkl'%self.id_t, 'wb') as f:
#            pickle.dump(uerror_history_lb, f)
#
#        with open('Data_flie/verrorl_value-%d.pkl'%self.id_t, 'wb') as f:
#            pickle.dump(verror_history_lb, f)

    def __Loadmat(self,fileName):

        data = scipy.io.loadmat(fileName)

        tt = data['t'].T[:,(self.id_t*tint):((self.id_t +1)*tint + 1)]
        print('t = ', tt)
        t = tt.flatten()[:,None]
        print('t.shape = ', t.shape)

        x = data['x'].T.flatten()[:,None]
#        Exact_u = np.real(data['Exact'].T)
        self.Exact = data['Exact']#.T
        print('Exact.shape= ', self.Exact.shape)
#        print('x= ', x)

        self.Exact_u = self.Exact.real.T
        X, T = np.meshgrid(x, t)
        self.x=x
        self.t=t
        self.X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

        u_star_ori=self.Exact_u[:, (self.id_t*tint):((self.id_t +1)*tint + 1)]

        print('u_star_ori.shape= ', u_star_ori.shape)

        self.u_star=u_star_ori#.flatten()[:,None]
        print('self.u_star.shape= ', (self.u_star).shape)

        # Domain bounds
        lb = self.X_star.min(0)
        ub = self.X_star.max(0)

        #grab training points from domain
        print('x.shape[0]= ', x.shape[0])
        idx_x = np.random.choice(x.shape[0], N0, replace=False)
        self.idx_x=idx_x
        x0 = x[idx_x,:]

#        self.u0 = tf.cast(self.u_star[idx_x,0:1], dtype = tfdoubstr)
        if self.id_t==0:
            self.u0 = torch.tensor(self.Exact_u[idx_x, 0:1], dtype=torch.float32).cuda()
        else:
            u0 = np.array(self.u0_t).flatten()[:,None]
            self.u0 = torch.tensor(u0[idx_x,0:1], dtype=torch.float32).cuda()
        print('self.u0.shape= ', (self.u0).shape)
 
        idx_t = np.random.choice(t.shape[0], Nb, replace=False)
        self.idx_t=idx_t
        tb = t[idx_t,:]
        
        X_f = lb + (ub-lb)*lhs(2, self.N_f)
        self.x_f = torch.tensor(X_f[:, 0:1]).float().requires_grad_(True).cuda()
        self.t_f = torch.tensor(X_f[:, 1:2]).float().requires_grad_(True).cuda()
        
        X0 = np.concatenate((x0, 0*x0 + t[0]), 1) # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
        
        self.x0 = torch.tensor(X0[:, 0:1]).float().requires_grad_(True).cuda()
        self.t0 = torch.tensor(X0[:, 1:2]).float().requires_grad_(True).cuda()
        print('self.x0.shape= ', self.x0.shape)
        print('self.t0.shape= ', self.t0.shape)
        
        self.x_lb = torch.tensor(X_lb[:, 0:1]).float().requires_grad_(True).cuda()
        self.t_lb = torch.tensor(X_lb[:, 1:2]).float().requires_grad_(True).cuda()
        self.x_ub = torch.tensor(X_ub[:, 0:1]).float().requires_grad_(True).cuda()
        self.t_ub = torch.tensor(X_ub[:, 1:2]).float().requires_grad_(True).cuda()

        u_lb_all = self.u_star[0,  :].flatten()[:,None]
        u_ub_all = self.u_star[-1, :].flatten()[:,None]
        u_lb= u_lb_all[idx_t]
        u_ub= u_ub_all[idx_t]
        self.u_lb = torch.tensor(u_lb).float().requires_grad_(True).cuda()
        self.u_ub = torch.tensor(u_ub).float().requires_grad_(True).cuda()
        self.lables_0=(self.u0)
        self.lables_lb=(self.u_lb)
        self.lables_ub=(self.u_ub)

#    def error_u(self):
#        u_pred, f_u_pred = self.predict()#self.u_model, self.X_star)
#        u_star = self.Exact_u.T[(self.id_t*tint):((self.id_t +1)*tint + 1),:]
#        u_star = u_star.flatten()[:, None]
#        
#        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
#        
#        return error_u
#
#
#    def predict(self):
#        X_star = torch.tensor(self.X_star, dtype=torch.float32, device=device, requires_grad=True)
#        u_star = self.u_model(X_star[:, 0:1], X_star[:, 1:2])
#
#        X_star = X_star.clone().detach().requires_grad_(True)
#        f_u_star = self.f_model(self, X_star[:, 0:1], X_star[:, 1:2])
#
#        u_star = u_star.detach().cpu().numpy()
#        f_u_star = f_u_star.detach().cpu().numpy()
#
#        return u_star, f_u_star

