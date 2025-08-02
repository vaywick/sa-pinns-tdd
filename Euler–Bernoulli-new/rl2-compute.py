import numpy as np
import pickle

# 网格点的数量
num_points_x = 151
num_points_t = 101

def uv(x,t):
    i=1j
    u= np.sin(x)*np.cos(np.pi*t)

    return u


# 坐标范围
x_min = 0
x_max = 8*np.pi
t_min = 0
t_max = 1

#uv(1,2)
# 生成坐标点
x = np.linspace(x_min, x_max, num_points_x).reshape(-1, 1)
t = np.linspace(t_min, t_max, num_points_t).reshape(-1, 1)

# 生成网格点
X, T = np.meshgrid(x, t)

# 计算函数值
Exact_u = np.vectorize(uv)(X, T)
Exact_r = Exact_u
cau_Exact_r = Exact_u

# 读取PKL文件
with open('comb_u_pred-causal-eb.pkl', 'rb') as f:
    u_pred = pickle.load(f).detach().numpy()

h_pred_r = u_pred

cau_theta_pred = h_pred_r.reshape((num_points_x, num_points_t)).T

cau_perror_theta = Exact_r - cau_theta_pred

with open('Data_flie/comb_u_pred.pkl', 'rb') as f:
    u_pred = pickle.load(f)

h_pred_r = u_pred

X, T = np.meshgrid(x, t)

U_pred = h_pred_r.reshape((num_points_x, num_points_t)).T

perror = Exact_r - U_pred
error_u = np.linalg.norm(perror, 2) / np.linalg.norm(Exact_r, 2)

print('relative-l2 error= ', error_u)

error_theta_causal = np.linalg.norm(cau_perror_theta, 2) / np.linalg.norm(cau_Exact_r, 2)

print('cau-relative-l2 error= ', error_theta_causal)


