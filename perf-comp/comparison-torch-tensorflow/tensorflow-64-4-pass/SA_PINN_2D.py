from SA_PINN import*
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

class SA_PINN_2D:
    @tf.function
    def u_x_model(self, x,y, t):
        u=self.model(tf.concat([x,y,t],1))
        
        return u
    def __init__(self, mat_filename, layers: [], tf_iter: int, newton_iter: int, f_model, ux_model=u_x_model, Loss=..., lbfgs_lr=0.5, N_f=10000, checkPointPath="./checkPoint"):
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
        u_weights1 = tf.Variable(tf.random.uniform([self.u0.shape[0], 1]))
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

    def fit(self):

        batch_sz = self.N_f
        n_batches =  self.N_f // batch_sz
        start_time = time.time()
        tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90)
        optimizers=[]
        for i in range(len(self.SA_weights)):
            optimizers.append(tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90))

        print("starting Adam training")

        for epoch in range(self.tf_iter):
            for i in range(n_batches):
                loss_value,mse_0, mse_f, grads, SA_grads = self.__grad(self.x_f,self.y_f,self.t_f,self.u0,self.u_x_lb,self.u_x_ub,self.u_y_lb,self.u_y_ub,self.XY,self.XT,self.YT,self.SA_weights)

                tf_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                a=0
                for key in self.SA_weights:
                    optimizers[a].apply_gradients(zip([-SA_grads[a]],[self.SA_weights[key]]))
                    a+=1
                a=0


            if (epoch+1) % 100 == 0:
                elapsed = time.time() - start_time
#                print('It: %d, Time: %.2f' % (epoch, elapsed))
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value))
#                tf.print(f"mse_0: {mse_0}  mse_f: {mse_f}   total loss: {loss_value}")
                start_time = time.time()

        print("Starting L-BFGS training")

        loss_and_flat_grad = self.__get_loss_and_flat_grad(self.x_f,self.y_f,self.t_f,self.u0,self.u_x_lb,self.u_x_ub,self.u_y_lb,self.u_y_ub,self.XY,self.XT,self.YT,self.SA_weights)
        lbfgs(self.model,loss_and_flat_grad, self.get_weights(self.model),
        Struct(), maxIter=self.newton_iter, learningRate=self.lbfgs_lr)

    def __grad(self, x_f_batch,y_f_batch, t_f_batch, u0,u_x_lb,u_x_ub,u_y_lb,u_y_ub, XY, XT, YT, SA_weights):
        with tf.GradientTape(persistent=True) as tape:
            #tape.watch(col_weights)
            #tape.watch(u_weights)
            loss_value, mse_0, mse_f = self.Loss(self,x_f_batch,y_f_batch, t_f_batch, u0,u_x_lb,u_x_ub,u_y_lb,u_y_ub, XY, XT, YT, SA_weights)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            #print(grads)
            SA_grads=[]
            for key in SA_weights:
                SA_grads.append(tape.gradient(loss_value,SA_weights[key]))
            #grads_col = tape.gradient(loss_value, col_weights)
            #grads_u = tape.gradient(loss_value, u_weights)

        return loss_value, mse_0, mse_f, grads, SA_grads

    def __get_loss_and_flat_grad(self,x_f_batch,y_f_batch, t_f_batch, u0,u_x_lb,u_x_ub,u_y_lb,u_y_ub, XY, XT, YT, SA_weights):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.__set_weights(self.model, w, self.sizes_w, self.sizes_b)
                loss_value, _, _ = self.Loss(self,x_f_batch,y_f_batch, t_f_batch, u0,u_x_lb,u_x_ub,u_y_lb,u_y_ub, XY, XT, YT, SA_weights)
            grad = tape.gradient(loss_value, self.model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            #print(loss_value, grad_flat)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def __Loadmat(self, fileName):
        #return super().__Loadmat(fileName)
        data = scipy.io.loadmat(fileName)

        t = data['t'].flatten()[:,None]
        x = data['x'].flatten()[:,None]
        y = data['y'].flatten()[:,None]
        self.x=x
        self.y=y
        self.t=t
        ub=np.array([x.min(0),y.min(0),t.min(0)]).T
        lb=np.array([x.max(0),y.max(0),t.max(0)]).T
        X_f = lb + ((ub-lb)*lhs(3, self.N_f))

        XY_X,XY_Y=np.meshgrid(x,y)#XY平面上的点的x，y坐标
        XT_X,XT_T=np.meshgrid(x,t)
        YT_Y,YT_T=np.meshgrid(y,t)
        XY=np.hstack((XY_X.flatten()[:, None], XY_Y.flatten()[:, None]))
        XT=np.hstack((XT_X.flatten()[:, None], XT_T.flatten()[:, None]))
        YT=np.hstack((YT_Y.flatten()[:, None], YT_T.flatten()[:, None]))

#        u0=data['Exact_u0'].flatten()[:,None]
#        u_x_lb=data['Exact_u_x_lb'].flatten()[:,None]
#        u_x_ub=data['Exact_u_x_ub'].flatten()[:,None]
#        u_y_lb=data['Exact_u_y_lb'].flatten()[:,None]
#        u_y_ub=data['Exact_u_y_ub'].flatten()[:,None]

        selected_indices = np.random.choice(XY.shape[0], N0, replace=False)
        selected_indicesxt = np.random.choice(XT.shape[0], Nb, replace=False)
        selected_indicesyt = np.random.choice(YT.shape[0], Nb, replace=False)
        XY= XY[selected_indices, :]
        YT= YT[selected_indicesyt, :]
        XT= XT[selected_indicesxt, :]

        u0all=data['Exact_u0'].flatten()[:,None]
        u0= u0all[selected_indices]
        u_x_lball =data['Exact_u_x_lb'].flatten()[:,None]
        u_x_uball =data['Exact_u_x_ub'].flatten()[:,None]
        u_y_lball =data['Exact_u_y_lb'].flatten()[:,None]
        u_y_uball =data['Exact_u_y_ub'].flatten()[:,None]

        u_x_lb= u_x_lball[selected_indicesyt]
        u_x_ub= u_x_uball[selected_indicesyt]
        u_y_lb= u_y_lball[selected_indicesxt]
        u_y_ub= u_y_uball[selected_indicesxt]


        self.u0=tf.cast(u0,dtype=tfdoubstr)
        self.u_x_lb=tf.cast(u_x_lb,dtype=tfdoubstr)
        self.u_x_ub=tf.cast(u_x_ub,dtype=tfdoubstr)
        self.u_y_lb=tf.cast(u_y_lb,dtype=tfdoubstr)
        self.u_y_ub=tf.cast(u_y_ub,dtype=tfdoubstr)
        self.XY=tf.convert_to_tensor(XY,dtype=tfdoubstr)
        self.XT=tf.convert_to_tensor(XT,dtype=tfdoubstr)
        self.YT=tf.convert_to_tensor(YT,dtype=tfdoubstr)
        self.x_f=tf.convert_to_tensor(X_f[:,0:1],dtype=tfdoubstr)
        self.y_f=tf.convert_to_tensor(X_f[:,1:2],dtype=tfdoubstr)
        self.t_f=tf.convert_to_tensor(X_f[:,2:3],dtype=tfdoubstr)
        self.t0=tf.cast(tf.reshape(tf.repeat(t.min(0),XY[:,1:2].shape[0]),(XY[:,1:2].shape[0],-1)),dtype=tfdoubstr)
        self.x_lb=tf.cast(tf.reshape(tf.repeat(x.min(0),YT[:,1:2].shape[0]),(YT[:,1:2].shape[0],-1)),dtype=tfdoubstr)
        self.x_ub=tf.cast(tf.reshape(tf.repeat(x.max(0),YT[:,1:2].shape[0]),(YT[:,1:2].shape[0],-1)),dtype=tfdoubstr)
        self.y_lb=tf.cast(tf.reshape(tf.repeat(y.min(0),XT[:,1:2].shape[0]),(XT[:,1:2].shape[0],-1)),dtype=tfdoubstr)
        self.y_ub=tf.cast(tf.reshape(tf.repeat(y.max(0),XT[:,1:2].shape[0]),(XT[:,1:2].shape[0],-1)),dtype=tfdoubstr)


        X,Y,T=np.meshgrid(x,y,t)
        self.X_star=np.hstack((X.flatten()[:, None],Y.flatten()[:, None],T.flatten()[:, None]))
        self.u_star=data["Exact"].flatten()[:,None]

    def predict(self, x, y, t):
        X,Y,T=np.meshgrid(x,y,t)
        XX_star=np.hstack((X.flatten()[:, None],Y.flatten()[:, None],T.flatten()[:, None]))
        X_star = tf.convert_to_tensor(XX_star, dtype=tfdoubstr)
        u_star= self.u_x_model(X_star[:,0:1],
                        X_star[:,1:2],X_star[:,2:3])

        f_u_star = self.f_model(self,X_star[:,0:1],
                    X_star[:,1:2],X_star[:,2:3])

        return u_star.numpy(), f_u_star.numpy()
