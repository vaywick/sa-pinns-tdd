from SA_PINN import*

class SA_PINN2(SA_PINN):
    def uv_model(self,x,t):
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

#        X=torch.cat([x, t], dim=1)
        X = torch.cat([x, t], dim=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        uv = self.model(H)

#        uv = self.model(torch.cat([x, t], dim=1))
        u=uv[:,0:1]
        v=uv[:,1:2]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        return u,v,u_t,v_t

    def __init__(self, mat_filename, layers: [], adam_iter: int, newton_iter: int, f_model, Loss=..., lbfgs_lr=0.8, N_f=10000, id_t=0, u0_tmp=0,v0_tmp=0, checkPointPath="./checkPoint"):
        super().__init__(mat_filename, layers, adam_iter, newton_iter, f_model, Loss, lbfgs_lr, N_f, id_t, u0_tmp, checkPointPath)
        self.v0_tmp=v0_tmp
        self.__Loadmat(mat_filename)
        self.col_v_weights = nn.Parameter(torch.full((N_f, 1), 100.0, device=device))
        self.v_weights = nn.Parameter(torch.full((self.x0.shape[0], 1), 100.0, device=device))
        self.v_lb_weights = nn.Parameter(torch.full((self.x_lb.shape[0], 1), 100.0, device=device))
        self.v_ub_weights = nn.Parameter(torch.full((self.x_ub.shape[0], 1), 100.0, device=device))

    def __Loadmat(self, fileName):
        self.Exact_v=self.Exact.imag.T
        v_star_ori=self.Exact_v
        self.v_star=v_star_ori
        v_star_ori=self.Exact_v[:, (self.id_t*tint):((self.id_t +1)*tint + 1)]
        self.v_star=v_star_ori
        if self.id_t==0:
            self.v0 = torch.tensor(self.Exact_v[self.idx_x, 0:1], dtype=torch.float32).cuda()
            self.v0_t = torch.tensor(self.Exact_v_t[self.idx_x, 0:1], dtype=torch.float32).cuda()
        else:
            v0 = np.array(self.v0_t).flatten()[:,None]
            self.v0 = torch.tensor(v0[self.idx_x,0:1], dtype=torch.float32).cuda()

        v_lb_all = self.v_star[0,  :].flatten()[:,None]
        v_ub_all = self.v_star[-1, :].flatten()[:,None]
        v_lb= v_lb_all[self.idx_t]
        v_ub= v_ub_all[self.idx_t]
        self.v_lb = torch.tensor(v_lb).float().requires_grad_(True).cuda()
        self.v_ub = torch.tensor(v_ub).float().requires_grad_(True).cuda()


    def error_u(self):
#        u_pred,v_pred, f_u_pred, f_v_pred = self.predict()#self.u_model, self.X_star)
        X_star = torch.tensor(self.X_star, dtype=torch.float32, device=device, requires_grad=True)
        u_pred,v_pred,_, _ = self.uv_model(X_star[:, 0:1], X_star[:, 1:2])
        u_star = self.Exact_u.T[(self.id_t*tint):((self.id_t +1)*tint + 1),:]
        u_star = u_star.flatten()[:, None]
        
        v_star = self.Exact_v.T[(self.id_t*tint):((self.id_t +1)*tint + 1),:]
        v_star = v_star.flatten()[:, None]

        u_pred= u_pred.detach().cpu().numpy()
        v_pred= v_pred.detach().cpu().numpy()
        
        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
        error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
        
        return error_u, error_v

    def predict(self):
        X_star = torch.tensor(self.X_star, dtype=torch.float32, device=device, requires_grad=True)
        u_pred,v_pred,_, _ = self.uv_model(X_star[:, 0:1], X_star[:, 1:2])

        X_star = X_star.clone().detach().requires_grad_(True)
        f_u_pred,f_v_pred = self.f_model(self, X_star[:, 0:1], X_star[:, 1:2])

        u_pred = u_pred.detach().cpu().numpy()
        v_pred = v_pred.detach().cpu().numpy()
        f_u_pred = f_u_pred.detach().cpu().numpy()
        f_v_pred = f_v_pred.detach().cpu().numpy()

        return u_pred,v_pred, f_u_pred,f_v_pred
