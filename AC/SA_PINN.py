import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import math
import matplotlib.gridspec as gridspec
import pickle
import os
import datetime
#from plotting import newfig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from scipy.interpolate import griddata
#from eager_lbfgs import lbfgs, Struct
from pyDOE import lhs
from globalf import *

# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)


class SA_PINN:
    def __DefaultLoss(self,x_f_batch, t_f_batch,
             x0, t0, u0,u_lb,u_ub, x_lb,
             t_lb, x_ub, t_ub,SA_weight):
        f_u_pred = self.f_model(self,x_f_batch, t_f_batch)
        u0_pred = self.model(tf.concat([x0, t0],1))
        u_lb_pred, _ = self.u_x_model(x_lb, t_lb)
        u_ub_pred, _ = self.u_x_model(x_ub, t_ub)

        mse_0_u = tf.reduce_mean(tf.square(SA_weight["u_weights"]*(u0 - u0_pred)))

        mse_b_u = tf.reduce_mean(tf.square(u_lb_pred - u_lb)) + \
            tf.reduce_mean(tf.square(u_ub_pred - u_ub)) #since ub/lb is 0

        mse_f_u = tf.reduce_mean(tf.square(SA_weight["col_weights"]*f_u_pred))
            
        return  mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_f_u
        
    @tf.function
    def u_x_model(self,x,t):
        u = self.model(tf.concat([x,t],1))
        u_x = tf.gradients(u,x)
        return u,u_x

    def __init__(self,mat_filename,layers:[],tf_iter:int,newton_iter:int,f_model,ux_model=u_x_model,Loss=__DefaultLoss,lbfgs_lr=0.8,N_f=10000, id_t=0, u0_t= 0,checkPointPath="./checkPoint"):
        self.N_f=N_f
        self.u0_t= u0_t
        self.id_t= id_t
        self.__Loadmat(mat_filename)
        self.layers=layers
        self.sizes_w=[]
        self.sizes_b=[]
        self.lbfgs_lr=lbfgs_lr

        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

#        col_weights1 = tf.Variable(tf.reshape(tf.repeat(100.0, N_f),(N_f, -1)))
        col_weights1 = tf.Variable(tf.random.uniform([self.N_f, 1]))
#        u_weights1 = tf.Variable(tf.random.uniform([self.x0.shape[0], 1]))
##        u_weights1 = tf.Variable(100.*tf.random.uniform([self.x0.shape[0], 1]))
        u_weights1 = tf.Variable(tf.reshape(tf.repeat(100.0, self.x0.shape[0]),(self.x0.shape[0], -1)))
        col_weights = tf.cast(col_weights1, dtype=tfdoubstr)
        u_weights = tf.cast(u_weights1, dtype=tfdoubstr)
        self.col_weights=tf.Variable(col_weights)
        self.u_weights=tf.Variable(u_weights)
#        self.SA_weights={"u_weights":self.u_weights,"col_weights":self.col_weights}
        self.model=self.__neural_net(self.layers)
        self.model.summary()
        self.Loss=Loss
        self.tf_iter=tf_iter
        self.newton_iter=newton_iter
        self.f_model=f_model
        self.ux_model=ux_model
        self.checkPointPath=checkPointPath
        if not os.path.exists(self.checkPointPath):
            os.makedirs(self.checkPointPath)


    def __Loadmat(self,fileName):

        data = scipy.io.loadmat(fileName)

        tt = data['tt'][:,(self.id_t*tint):((self.id_t +1)*tint + 1)]
        print('tt.shape= ', tt.shape)
        t = tt.flatten()[:,None]
        print('t.shape = ', t.shape)
#        print('t = ', t)

        x = data['x'].flatten()[:,None]
        Exact = data['uu']
#        self.Exact = data['uu']
        self.Exact = np.real(data['uu'])
        print('Exact.shape= ', Exact.shape)
#        print('x= ', x)

        self.Exact_u = np.real(Exact)
        X, T = np.meshgrid(x, t)
        self.x=x
        self.t=t
        self.X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # X_star.shape: (103113, 2)
#        self.u_star = Exact.flatten()[:, None]  # u_star.shape: (103113, 1)

        u_star_ori=Exact[:, (self.id_t*tint):((self.id_t +1)*tint + 1)]

        print('u_star_ori.shape= ', u_star_ori.shape)

#        if(self.id_t!=0):
##           print('self.u0_t.shape= ', self.u0_t.shape)
##           print('self.u0_t.shape= ', self.u0_t)
#           u_star_ori[:,0] = self.u0_t
#           print('u_star_ori.shape= ', u_star_ori.shape)

#        self.u_star=u_star_ori.flatten()[:,None]
        self.u_star=u_star_ori#.flatten()[:,None]
        print('self.u_star.shape= ', (self.u_star).shape)

        # Domain bounds
        lb = self.X_star.min(0)#下界
        ub = self.X_star.max(0)#上界

        #grab training points from domain
        print('x.shape[0]= ', x.shape[0])
        idx_x = np.random.choice(x.shape[0], N0, replace=False)
        x0 = x[idx_x,:]
        print('x0.shape= ', x0.shape)
        self.u0 = tf.cast(self.Exact_u[idx_x,0:1], dtype = tfdoubstr)

#        self.u0 = tf.cast(self.u_star[idx_x,0:1], dtype = tfdoubstr)
        if self.id_t==0:
            self.u0 = tf.cast(self.u_star[idx_x, 0:1], dtype = tfdoubstr)
        else:
#            self.u0=u0
#            print('u0_t.shape= ', u0_t.shape)
            u0 = np.array(self.u0_t).flatten()[:,None]
            self.u0 = tf.cast(u0[idx_x,0:1], dtype = tfdoubstr)
        print('self.u0.shape= ', (self.u0).shape)
#        print('self.u0= ', self.u0)
 
        idx_t = np.random.choice(t.shape[0], Nb, replace=False)
        tb = t[idx_t,:]
        print('tb.shape= ', tb.shape)
        
        # Grab collocation points using latin hpyercube sampling
        
        X_f = lb + (ub-lb)*lhs(2, self.N_f)
        
        self.x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tfdoubstr)
        self.t_f = tf.convert_to_tensor(X_f[:,1:2], dtype=tfdoubstr)
        print('self.x_f.shape= ', self.x_f.shape)
        print('self.t_f.shape= ', self.t_f.shape)
        
        X0 = np.concatenate((x0, 0*x0 + t[0]), 1) # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
        
        self.x0 = tf.cast(X0[:,0:1], dtype = tfdoubstr)
        self.t0 = tf.cast(X0[:,1:2], dtype = tfdoubstr)
        print('self.x0.shape= ', self.x0.shape)
        print('self.t0.shape= ', self.t0.shape)
        
        self.x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tfdoubstr)
        self.t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tfdoubstr)
        
        self.x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tfdoubstr)
        self.t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tfdoubstr)

        u_lb_all = self.u_star[0,  :].flatten()[:,None]
        u_ub_all = self.u_star[-1, :].flatten()[:,None]
        u_lb= u_lb_all[idx_t]
        u_ub= u_ub_all[idx_t]
        self.u_lb=tf.cast(u_lb,dtype=tfdoubstr)
        self.u_ub=tf.cast(u_ub,dtype=tfdoubstr)


    def __set_weights(self,model, w, sizes_w, sizes_b):
        for i, layer in enumerate(model.layers[0:]):
            start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
            end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(sizes_w[i] / sizes_b[i])
            weights = tf.reshape(weights, [w_div, sizes_b[i]])
            biases = w[end_weights:end_weights + sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)
    
    def get_weights(self,model):
        w = []
        for layer in model.layers[0:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)

        w = tf.convert_to_tensor(w)
        return w

    def __neural_net(self,layer_sizes):
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
        for width in layer_sizes[1:-1]:
            model.add(layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal"))
        model.add(layers.Dense(
            layer_sizes[-1], activation=None,
            kernel_initializer="glorot_normal"))
        return model
   
    @tf.function
    def __grad(self, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch,x_lb, t_lb, x_ub, t_ub,col_weights,u_weights):
        with tf.GradientTape(persistent=True) as tape:
            loss_value, mse_0, mse_f = self.Loss(self,x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch, x_lb, t_lb, x_ub, t_ub,col_weights,u_weights)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            #print(grads)
            grads_col = tape.gradient(loss_value, col_weights)
            grads_u = tape.gradient(loss_value, u_weights)

        return loss_value, mse_0, mse_f, grads, grads_col, grads_u
   
    def fit(self):
        # Built in support for mini-batch, set to N_f (i.e. full batch) by default
        batch_sz = self.N_f
        n_batches =  self.N_f // batch_sz
        start_time = time.time()
        tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90)
#        tf_optimizer_coll = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90)
#        tf_optimizer_u = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90)
        tf_optimizer_weights = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90)

        

        print("starting Adam training")

        lossad_all = []
        error_all = []
        error_all_v = []
        for epoch in range(self.tf_iter):
            for i in range(n_batches):

                x0_batch = self.x0#[i*batch_sz:(i*batch_sz + batch_sz),]
                t0_batch = self.t0#[i*batch_sz:(i*batch_sz + batch_sz),]
                u0_batch = self.u0#[i*batch_sz:(i*batch_sz + batch_sz),]
                u_lb_batch=self.u_lb#[i*batch_sz:(i*batch_sz + batch_sz),]
                u_ub_batch=self.u_ub#[i*batch_sz:(i*batch_sz + batch_sz),]

                x_f_batch = self.x_f#[i*batch_sz:(i*batch_sz + batch_sz),]
                t_f_batch = self.t_f#[i*batch_sz:(i*batch_sz + batch_sz),]

                loss_value,mse_0, mse_f, grads, grads_col, grads_u = self.__grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch,self.x_lb, self.t_lb, self.x_ub, self.t_ub, self.col_weights,self.u_weights)

                tf_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                tf_optimizer_weights.apply_gradients(zip([-grads_col, -grads_u], [self.col_weights, self.u_weights]))

            if (epoch+1) % 100 == 0:
                error_u_value = self.error_u()

                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e, error: %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value, error_u_value))
                start_time = time.time()


        #l-bfgs-b optimization
        print("Starting L-BFGS training")

        loss_and_flat_grad = self.__get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch, self.x_lb, self.t_lb, self.x_ub, self.t_ub, self.col_weights, self.u_weights)

        self.lbfgs(self.checkPointPath,self.model,loss_and_flat_grad,
        self.get_weights(self.model),
        Struct(), maxIter=self.newton_iter, learningRate=self.lbfgs_lr, id_t=self.id_t)


# L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
    def __get_loss_and_flat_grad(self,x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.__set_weights(self.model, w, self.sizes_w, self.sizes_b)
                loss_value, _, _ = self.Loss(self,x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch, x_lb, t_lb, x_ub, t_ub, self.col_weights,self.u_weights)
            grad = tape.gradient(loss_value, self.model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            #print(loss_value, grad_flat)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def error_u(self):
        X_star = tf.convert_to_tensor(self.X_star, dtype=tfdoubstr)
        u_pred,_ = self.u_x_model(X_star[:, 0:1], X_star[:, 1:2])
        u_star = self.Exact_u.T[(self.id_t*tint):((self.id_t +1)*tint + 1),:]
        u_star = u_star.flatten()[:, None]

        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

        return error_u


    def predict(self):
        X_star = tf.convert_to_tensor(self.X_star, dtype=tfdoubstr)
        u_star, _ = self.u_x_model(X_star[:,0:1],
                        X_star[:,1:2])

        f_u_star = self.f_model(model=self,x=X_star[:,0:1],
                    t=X_star[:,1:2])

        return u_star.numpy(), f_u_star.numpy()

    def dot(self, a, b):
      """Dot product function since TensorFlow doesn't have one."""
      return tf.reduce_sum(a*b)
    
    def verbose_func(self, s):
      print(s)
    
    final_loss = None
    times = []
    def lbfgs(self, fileName,u_model,opfunc, x, state, maxIter = 100, learningRate = 1, do_verbose = True, id_t=0):
      """port of lbfgs.lua, using TensorFlow eager mode.
      """
    
      global final_loss, times
      
      maxEval = maxIter*1.25
      tolFun = 1e-7
      tolX = 1e-11
      nCorrection = 50
      isverbose = False
    
      # verbose function
      if isverbose:
        verbose = self.verbose_func
      else:
        verbose = lambda x: None
        
      f, g = opfunc(x)
    
      f_hist = [f]
      currentFuncEval = 1
      state.funcEval = state.funcEval + 1
      p = g.shape[0]
    
      # check optimality of initial point
      tmp1 = tf.abs(g)
      if tf.reduce_sum(tmp1) <= tolFun:
        verbose("optimality condition below tolFun")
        return x, f_hist
    
      # optimize for a max of maxIter iterations
      nIter = 0
      times = []
    
      loss_l = []
      error_l = []
      error_l_v = []
    
      start_time = time.time()
      while nIter < maxIter:
        
        # keep track of nb of iterations
        nIter = nIter + 1
        state.nIter = state.nIter + 1
    
        ############################################################
        ## compute gradient descent direction
        ############################################################
        if state.nIter == 1:
          d = -g
          old_dirs = []
          old_stps = []
          Hdiag = 1
        else:
          # do lbfgs update (update memory)
          y = g - g_old
          s = d*t
          ys = self.dot(y, s)
          
          if ys > 1e-10:
            # updating memory
            if len(old_dirs) == nCorrection:
              # shift history by one (limited-memory)
              del old_dirs[0]
              del old_stps[0]
    
            # store new direction/step
            old_dirs.append(s)
            old_stps.append(y)
    
            # update scale of initial Hessian approximation
            Hdiag = ys/self.dot(y, y)
    
          # compute the approximate (L-BFGS) inverse Hessian 
          # multiplied by the gradient
          k = len(old_dirs)
    
          # need to be accessed element-by-element, so don't re-type tensor:
          ro = [0]*nCorrection
          for i in range(k):
            ro[i] = 1/self.dot(old_stps[i], old_dirs[i])
            
    
          # iteration in L-BFGS loop collapsed to use just one buffer
          # need to be accessed element-by-element, so don't re-type tensor:
          al = [0]*nCorrection
    
          q = -g
          for i in range(k-1, -1, -1):
            al[i] = self.dot(old_dirs[i], q) * ro[i]
            q = q - al[i]*old_stps[i]
    
          # multiply by initial Hessian
          r = q*Hdiag
          for i in range(k):
            be_i = self.dot(old_stps[i], r) * ro[i]
            r += (al[i]-be_i)*old_dirs[i]
            
          d = r
          # final direction is in r/d (same object)
    
        g_old = g
        f_old = f
        
        ############################################################
        ## compute step length
        ############################################################
        # directional derivative
        gtd = self.dot(g, d)
    
        # check that progress can be made along that direction
        if gtd > -tolX:
          verbose("Can not make progress along direction.")
          break
    
        # reset initial guess for step size
        if state.nIter == 1:
          tmp1 = tf.abs(g)
          t = min(1, 1/tf.reduce_sum(tmp1))
        else:
          t = learningRate
    
    
        x += t*d
    
        if nIter != maxIter:
        # re-evaluate function only if not in last iteration
        # the reason we do this: in a stochastic setting,
        # no use to re-evaluate that function here
          f, g = opfunc(x)
    
        lsFuncEval = 1
        f_hist.append(f)
    
    
        # update func eval
        currentFuncEval = currentFuncEval + lsFuncEval
        state.funcEval = state.funcEval + lsFuncEval
    
        ############################################################
        ## check conditions
        ############################################################
        if nIter == maxIter:
          break
    
        if currentFuncEval >= maxEval:
          # max nb of function evals
          print('max nb of function evals')
          break
    
        tmp1 = tf.abs(g)
        if tf.reduce_sum(tmp1) <=tolFun:
          # check optimality
          print('optimality condition below tolFun')
          break
        
        tmp1 = tf.abs(d*t)
        if tf.reduce_sum(tmp1) <= tolX:
          # step size below tolX
          print('step size below tolX')
          break
        
        if tf.abs(f,f_old) < tolX:
          # function value changing less than tolX
          print('function value changing less than tolX'+str(tf.abs(f-f_old)))
          break
    

        if do_verbose:
          if (nIter+1) % 100 == 0:
            error_u_value = self.error_u()

            elapsed = time.time() - start_time
            print("Step: %3d time: %.2f loss: %.4e error u: %.4e"%(nIter+1, elapsed, f.numpy(), error_u_value))
            start_time = time.time()
    
    
        if nIter == maxIter - 1:
          final_loss = f.numpy()
    
    
      # save state
      state.old_dirs = old_dirs
      state.old_stps = old_stps
      state.Hdiag = Hdiag
      state.g_old = g_old
      state.f_old = f_old
      state.t = t
      state.d = d
    
      return x, f_hist, currentFuncEval
    
