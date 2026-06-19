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
from eager_lbfgs import lbfgs, Struct
from pyDOE import lhs
from globalf import *

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
            tf.reduce_mean(tf.square(u_ub_pred - u_ub))

        mse_f_u = tf.reduce_mean(tf.square(SA_weight["col_weights"]*f_u_pred))
            
        return  mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_f_u
        
    
    @tf.function
    def u_x_model(self,x,t):
        u = self.model(tf.concat([x,t],1))
        u_x = tf.gradients(u,x)
        return u,u_x

    def __init__(self,mat_filename,layers:[],tf_iter:int,newton_iter:int,f_model,ux_model=u_x_model,Loss=__DefaultLoss,lbfgs_lr=0.5,N_f=10000,checkPointPath="./checkPoint"):
        self.N_f=N_f
        self.__Loadmat(mat_filename)
        self.layers=layers
        self.sizes_w=[]
        self.sizes_b=[]
        self.lbfgs_lr=lbfgs_lr

        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

        col_weights1 = tf.Variable(tf.reshape(tf.repeat(100.0, N_f),(N_f, -1)))
        u_weights1 = tf.Variable(tf.random.uniform([self.x0.shape[0], 1]))
        col_weights = tf.cast(col_weights1, dtype=tfdoubstr)
        u_weights = tf.cast(u_weights1, dtype=tfdoubstr)
        self.col_weights=tf.Variable(col_weights)
        self.u_weights=tf.Variable(u_weights)
        self.SA_weights={"u_weights":self.u_weights,"col_weights":self.col_weights}
        self.model=self.__neural_net(self.layers)
        self.model.summary()
        self.Loss=Loss
        self.tf_iter=tf_iter
        self.newton_iter=newton_iter
        self.f_model=f_model
        self.ux_model=ux_model
    
    def __Loadmat(self,fileName):

        data = scipy.io.loadmat(fileName)

        t = data['t'].flatten()[:,None]
        x = data['x'].flatten()[:,None]
        Exact = data['Exact']
        self.Exact_u = np.real(Exact)
        X, T = np.meshgrid(x, t)
        self.x=x
        self.t=t
        self.X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.u_star = Exact.flatten()[:, None]

        # Domain bounds
        lb = self.X_star.min(0)#下界
        ub = self.X_star.max(0)#上界
        #grab random points off the initial condition
        #idx_x = np.random.choice(x.shape[0], N0, replace=False)
        #self.x0 = x
        #u0 = tf.cast(Exact_u[idx_x,0:1], dtype = tfdoubstr)#初始条件
        self.u0 = tf.cast(self.Exact_u[0:1, :].T, dtype = tfdoubstr)
        #u0 = tf.cast(Exact_u[-1,idx_x], dtype = tfdoubstr)

        #idx_t = np.random.choice(t.shape[0], N_b, replace=False)
        #tb = t

        self.u_lb=tf.cast(self.Exact_u[:, 0:1],dtype=tfdoubstr)
        self.u_ub=tf.cast(self.Exact_u[:, -1:],dtype=tfdoubstr)
        # Sample collocation points via LHS
        X_f = lb + (ub-lb)*lhs(2, self.N_f)

        self.x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tfdoubstr)
        self.t_f = tf.convert_to_tensor(X_f[:,1:2], dtype=tfdoubstr)
        X0 =np.hstack((X[0:1, :].T, T[0:1, :].T))
        #print(X0)
        self.x0 = tf.cast(X0[:,0:1], dtype = tfdoubstr)
        self.t0 = tf.cast(X0[:,1:2], dtype = tfdoubstr)

        X_lb = np.hstack((X[:, 0:1], T[:, 0:1])) # (lb[0], tb)
        X_ub = np.hstack((X[:, -1:], T[:, -1:])) # (ub[0], tb)
        self.x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tfdoubstr)
        self.t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tfdoubstr)

        self.x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tfdoubstr)
        self.t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tfdoubstr)

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
    def __grad(self, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch,x_lb, t_lb, x_ub, t_ub, SA_weights):
        with tf.GradientTape(persistent=True) as tape:
            #tape.watch(col_weights)
            #tape.watch(u_weights)
            loss_value, mse_0, mse_f = self.Loss(self,x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch, x_lb, t_lb, x_ub, t_ub, SA_weights)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            #print(grads)
            SA_grads=[]
            for key in SA_weights:
                SA_grads.append(tape.gradient(loss_value,SA_weights[key]))

        return loss_value, mse_0, mse_f, grads, SA_grads
   
    def fit(self):

        batch_sz = self.N_f
        n_batches =  self.N_f // batch_sz
        start_time = time.time()
        tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90)
        tf_optimizer_coll = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90)
        tf_optimizer_u = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90)
        optimizers=[]
        for i in range(len(self.SA_weights)):
            optimizers.append(tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90))

        print("starting Adam training")

        for epoch in range(self.tf_iter):
            for i in range(n_batches):

                x0_batch = self.x0
                t0_batch = self.t0
                u0_batch = self.u0
                u_lb_batch=self.u_lb
                u_ub_batch=self.u_ub

                x_f_batch = self.x_f
                t_f_batch = self.t_f

                loss_value,mse_0, mse_f, grads, SA_grads = self.__grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch,self.x_lb, self.t_lb, self.x_ub, self.t_ub, self.SA_weights)

                tf_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                a=0
                for key in self.SA_weights:
                    optimizers[a].apply_gradients(zip([-SA_grads[a]],[self.SA_weights[key]]))
                    a+=1
                a=0


            if epoch % 10 == 0:
                elapsed = time.time() - start_time
#                print('It: %d, Time: %.2f' % (epoch, elapsed))
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e' % (epoch, elapsed, mse_0, mse_f, loss_value))
#                tf.print(f"mse_0: {mse_0}  mse_f: {mse_f}   total loss: {loss_value}")
                start_time = time.time()
        
        print("Starting L-BFGS training")

        loss_and_flat_grad = self.__get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch, self.x_lb, self.t_lb, self.x_ub, self.t_ub, self.col_weights, self.u_weights)

        lbfgs(self.checkPointPath,self.model,loss_and_flat_grad,
        self.get_weights(self.model),
        Struct(), maxIter=self.newton_iter, learningRate=self.lbfgs_lr)

# L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
    def __get_loss_and_flat_grad(self,x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.__set_weights(self.model, w, self.sizes_w, self.sizes_b)
                loss_value, _, _ = self.Loss(self,x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch,u_lb_batch,u_ub_batch, x_lb, t_lb, x_ub, t_ub, self.SA_weights)
            grad = tape.gradient(loss_value, self.model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            #print(loss_value, grad_flat)
            return loss_value, grad_flat

        return loss_and_flat_grad


    def predict(self):
        X_star = tf.convert_to_tensor(self.X_star, dtype=tfdoubstr)
        u_star, _ = self.u_x_model(X_star[:,0:1],
                        X_star[:,1:2])

        f_u_star = self.f_model(model=self,x=self.X_star[:,0:1],
                    t=self.X_star[:,1:2])

        return u_star.numpy(), f_u_star.numpy()
