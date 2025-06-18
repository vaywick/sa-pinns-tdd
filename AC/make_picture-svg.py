# import tensorflow as tf
import pickle
import numpy as np
import scipy.io
from scipy.interpolate import griddata
#from pyDOE import lhs
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib.ticker as ticker
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import sinh, cosh, atan, exp

#原始通过.mat文件获取数据
# 调用CPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 获取数据
# data = scipy.io.loadmat('sG_so2_data.mat')
# t = data['t'].flatten()[:,None]  # t.shape: (201, 1)
# x = data['x'].flatten()[:,None]  # x.shape: (513, 1)
# Exact = np.real(data['Exact'])  # Exact.shape: (201, 513)
# def u(x0, t0):
#     rw1=1
#     if(rw1==1):
#         Exact = 4*(t0**2 - x0**2 + 1)/(t0**2 + x0**2 + 1)**2
#     else:
#         Exact = 4*(-78125 - 2268*t0**6*x0**2 - 2430*t0**4*x0**4 + 1620*t0**2*x0**6 + 138750*t0**4 - 15750*x0**6 +25470*t0**6 \
#             + 109375*t0**2 + 265625*x0**2 + 33750*x0**2*t0**4 + 47250*x0**4*t0**2 + 382500*t0**2*x0**2 + 93750*x0**4 \
#             + 3807*t0**8 -2025*x0**8 + 243*t0**10 - 243*x0**10 + 729*t0**8*x0**2 + 486*t0**6*x0**4 - 486*t0**4*x0**6 - 729*t0**2*x0**8) \
#             / (625 + 475*t0**2 - 125*x0**2 + 51*t0**4 + 270*t0**2*x0**2 + 75*x0**4 + 9*t0**6 + 27*x0**2*t0**4 + 27*x0**4*t0**2 + 9*x0**6)**2

#     return Exact

#data = np.load('u_pred_list.npy-9.npy',allow_pickle=True) # (10, 512, 25)
#data = np.hstack((data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9]))
data2 = scipy.io.loadmat('AC.mat')
usol = data2['uu']
#Exact = usol[:,:-1]
Exact = usol.T

print('Exact.shape= ', Exact.shape)
#data = data.T

# 网格点的数量
num_points_x = 512
num_points_t = 201

#u_pred = scipy.io.loadmat('../burgers_shock.mat')['usol']
#data=Exact.T
with open('Data_flie/comb_u_pred.pkl', 'rb') as f:
    U_pred = pickle.load(f)

#U_pred = U_pred.reshape((num_points_t, num_points_x))
data = U_pred.reshape((num_points_x, num_points_t))
data = data.T
print('data.shape= ', data.shape)
#exit()


# 坐标范围
x_min = -1
x_max = 1
t_min = 0
t_max = 1

# 生成坐标点
x = np.linspace(x_min, x_max, num_points_x).reshape(-1, 1)
t = np.linspace(t_min, t_max, num_points_t).reshape(-1, 1)

# 生成网格点
X, T = np.meshgrid(x, t)

# 计算函数值
# Exact = np.vectorize(u)(X, T)

# 读取PKL文件
#with open('Data_flie-k-0.7/data_Y_Full_2nd.pkl', 'rb') as f:
#with open('Data_flie/data_Y_Full_2nd.pkl', 'rb') as f:
# with open('Data_flie/data_v.pkl', 'rb') as f:
#     U_pred = pickle.load(f)
# X, T = np.meshgrid(x, t)

# 绘图
def figsize(scale, nplots = 1):
    fig_width_pt = 390.0          # 这是一个浮点数，表示图形的宽度，以磅（points）为单位。Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # 表示将磅（points）转换为英寸（inches）的比例。Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # 黄金比。Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # 表示图形的宽度，通过将磅（points）转换为英寸（inches）并乘以一个缩放因子scale得到。width in inches
    fig_height = nplots*fig_width*golden_mean       # 通过将图形宽度乘以黄金比例，并乘以一个绘图数量nplots，得到图形的高度。height in inches
    fig_size = [fig_width,fig_height]               # 表示图形的尺寸，宽度在第一个元素，高度在第二个元素。
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
    "pgf.preamble": r"\usepackage[utf8x]{inputenc}\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
    }
mpl.rcParams.update(pgf_with_latex)
# 通过这些参数配置，Matplotlib将使用LaTeX来渲染文本和数学公式，并且图形的字体、尺寸和布局将符合LaTeX的默认样式。
# 这有助于生成与文档一致的高质量图形。


# I make my own newfig and savefig functions
def newfig(width, nplots = 1):
    # nplots = 1是newfig函数的默认参数值。当调用newfig函数时，如果不提供nplots参数的值，那么将使用默认值1。
    fig = plt.figure(figsize=figsize(width, nplots))  # 使用之前定义的figsize()函数计算出图形的尺寸
    ax = fig.add_subplot(111)  # 使用fig.add_subplot()函数创建一个新的轴对象。这里使用参数111表示在图形中创建一个包含单个子图的轴。
    # 111是一个特殊的参数值，它表示图形中只有一个轴，并且该轴占据整个图形的区域。
    return fig, ax

def savefig(filename, crop = True):
    if crop == True:
#        plt.savefig('{}.pgf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
#        plt.savefig('{}.pgf'.format(filename))
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))


#fig, ax = newfig(1.0, 1.1)
#fig, ax = newfig(1.1, 1.)
#fig, ax = newfig(2.3, .8)
fig, ax = newfig(1.3, 1.)
ax.axis('off')  # ax.axis('off')是一个Matplotlib函数调用，用于关闭轴的显示。它将隐藏轴上的刻度线、标签和边界框，使轴在图形中不可见

# 创建GridSpec对象
gs = gridspec.GridSpec(2, 3)

####### Row 0: u(t,x) ##################
########     Exact     ###########
# gs0 = gridspec.GridSpec(1, 3)
# gs0.update(top=1 - 0.06, bottom=0.7, left=0.1, right=0.9, wspace=0.5)
# ax = plt.subplot(gs0[:, :])
ax = fig.add_subplot(gs[0, 0])
# ax = plt.subplot(gs0[:, 0])
h = ax.imshow(Exact.T, interpolation='bilinear', cmap='rainbow',  # 注意这里的Exact.T
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
# Exact.T：这是要显示的二维数组，通过.T进行转置，以使得数组的行对应于y轴，列对应于x轴。转置是为了与网格布局中的坐标系对齐
# 'nearest'表示使用最近邻插值，即使用最接近的像素值进行绘制。这将导致图像显示较为锐利的边缘
# 'rainbow'表示使用彩虹颜色映射，其中不同的数值范围对应不同的颜色。这将帮助可视化数据的变化趋势
# extent=[t.min(), t.max(), x.min(), x.max()]：指定图像的坐标范围。t.min()和t.max()表示t轴的最小值和最大值，
# x.min()和x.max()表示x轴的最小值和最大值。这将根据数据的实际范围设置图像的坐标范围
# origin='lower'：指定图像坐标系的原点位置。'lower'表示原点位于图像的左下角。
# aspect='auto'：指定图像的宽高比。'auto'表示根据图形的尺寸自动调整宽高比，以保持图像不变形

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)  # colorbar指的就是在图最右边添加了色带
tick_locator = ticker.MaxNLocator(nbins=10)  # colorbar上的刻度值个数
cbar.locator = tick_locator
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
#cbar.set_ticks([-1, 0, 1])
#cbar.set_ticks([ -0.75, -0.5, -0.25, 0, 0.25,0.5,0.75])
cbar.update_ticks()

keep1 = 1 / 20
index1 = int(keep1 * t.shape[0])  # shape[0]表示矩阵的行数，相当于是在第1/8时间区间（从-2.5到2.5）长度处（大概是-1.875）
keep2 = 10 / 20
index2 = int(keep2 * t.shape[0])  # 相当于是在第3/8时间区间（从-2.5到2.5）长度处
keep3 = 19 / 20
#index3 = int(keep3 * t.shape[0])  # 相当于是在第7/8时间区间（从-2.5到2.5）长度处
index3 = int(keep3 * t.shape[0])  # 相当于是在第7/8时间区间（从-2.5到2.5）长度处

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[index1][0] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[index2] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[index3] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_yticks([-1,0,1])
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('$u(x,t)$-Reference(a)', fontsize=10)

U_pred = data.astype(np.float64)
#U_pred = U_pred.reshape((num_points_x, num_points_t))
#U_pred = U_pred.T
print('U_pred.shape= ', U_pred.shape)
#U_pred[-1,:] = U_pred[-2,:]
########  Predicted    ###########
ax = fig.add_subplot(gs[0, 1])
# ax = plt.subplot(gs0[:, 1])
#h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',  # 注意这里的u_pred.t
h = ax.imshow(U_pred.T, interpolation='bilinear', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)  # colorbar指的就是在图最右边添加了色带
#cbar.set_ticks([0.1, 0.2, 0.3, 0.4])
#cbar.set_ticks([0.2, 0.4, 0.6, 0.8, 1])
#cbar.set_ticks([-1, 0, 1])
#cbar.set_ticks([ -0.5, 0, 0.5])
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
#cbar.set_ticks([-0.75,-0.5,-0.25,0,0.25,0.5,0.75])

line = np.linspace(x.min(), x.max(), 2)[:, None]
#   numpy.linspace(start, end, num=num_points)将在start和end之间生成一个统一的序列，共有num_points个元素
# 画分割线
ax.plot(t[index1][0] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[index2] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[index3] * np.ones((2, 1)), line, 'w-', linewidth=1)
# t[index1][0] * np.ones((2, 1)) 生成一个包含两个相同值的数组，其中值为 t[index1][0]
# line 数组包含纵坐标的值。'w-' 指定线条的颜色为白色（'w'）以及直线的样式为实线（'-'）。linewidth=1 设置线条的宽度为1。

ax.set_yticks([-1,0,1])
ax.set_xlabel('$t$')
ax.set_title('$u(x,t)$-Predicted(b)', fontsize=10)

# 读取PKL文件
#with open('Data_flie-k-0.7/data_Y_Full_2nd.pkl', 'rb') as f:
#with open('Data_flie/data_Y_Full_2nd.pkl', 'rb') as f:
# with open('Data_flie/data_v.pkl', 'rb') as f:
#     U_pred = pickle.load(f)
# X, T = np.meshgrid(x, t)
#U_pred = U_pred.reshape((num_points_x, num_points_t))
#U_pred = U_pred.T

#U_pred[-1,:] = U_pred[-2,:]
print('Exact.shape= ', Exact.shape)
print('U_pred.shape= ', U_pred.shape)

perror = Exact - U_pred
error_u = np.linalg.norm(perror, 2) / np.linalg.norm(Exact, 2)

print('relative-l2 error= ', error_u)

#perror[-1, :] = perror[-2,:]
########      Error u(t,x)     ###########
ax = fig.add_subplot(gs[0, 2])
# ax = plt.subplot(gs0[:, 2])
h = ax.imshow(perror.T*1000, interpolation='bilinear', cmap='rainbow',
#h = ax.imshow(perror.T, interpolation='bilinear', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)
#cbar.set_ticks([0, 1])
#cbar.set_ticks([ -0.75, -0.5, -0.25, 0, 0.25,0.5,0.75])
cbar.set_ticks([ -2, 0, 2])

ax.set_yticks([-1,0,1])
ax.set_xlabel('$t$')
ax.set_title('Error(c)', fontsize = 10)
ax.text(1.02,1.02,r'$\times 10^{-3}$',fontsize=6)
#ax.text(5.1,42,r'$\times 10^{-1}$',fontsize=6)

# 调整子图之间的上下和左右间距
plt.subplots_adjust(hspace=0.45, wspace=0.45)


from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('xx-small')

####### Row 1: u(t,x) slices ##################
# gs1 = gridspec.GridSpec(1, 3)
# gs1.update(top=0.55, bottom=0.35, left=0.1, right=0.9, wspace=0.4)
# ax = plt.subplot(gs1[0, 0])
ax = fig.add_subplot(gs[1, 0])
ax.plot(x, Exact[index1, :], 'b-', linewidth=1, label='Exact')
ax.plot(x, U_pred[index1, :], 'r:', linewidth=1, label='Predicted')
ax.set_xlabel('$x$')  # x轴标签
ax.set_ylabel('$u(x,t)$')  # y轴标签
ax.set_title('$t = %.2f$(d)' % (t[index1][0]), fontsize=10)  # 标题
#ax.set_yticks([-6, -4, -2, 0]) # y轴刻度
#ax.set_xticks([ -10, -5, 0, 5, 10])
#ax.set_xticks([ -40, -20, 0, 20, 40])

# 添加线条注释
#legend = ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), prop={'size': 4})
#legend = ax.legend(loc='upper right',frameon=False, prop= fontP, bbox_to_anchor=(1., 0.92))# bbox_to_anchor=(1.0, 1.0), prop={'size': 4})
#legend = ax.legend(loc='upper right',frameon=False, prop= fontP)# bbox_to_anchor=(1.0, 1.0), prop={'size': 4})


# ax = plt.subplot(gs1[0, 1])
ax = fig.add_subplot(gs[1, 1])
ax.plot(x, Exact[index2, :], 'b-', linewidth=1, label='Exact')
ax.plot(x, U_pred[index2, :], 'r:', linewidth=1, label='Predicted')  # 虚线
ax.set_xlabel('$x$')  # x轴标签
ax.set_title('$t = %.2f$(e)' % (t[index2][0]), fontsize=10)
#ax.set_yticks([-7.0, -3.5, 0, 3.5, 7.0])
#ax.set_yticks([-6.0, -3, 0, 3., 6.])
#ax.set_xticks([ -10, -5, 0, 5, 10])
#ax.set_xticks([ -40, -20, 0, 20, 40])

# 添加线条注释
#legend = ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 4})
#legend = ax.legend(loc='upper right', frameon=False, prop= fontP)# bbox_to_anchor=(1.0, 1.0), prop={'size': 4})
# bbox_to_anchor=(1.0, 1.0),第一个值（1.0）表示图例的水平位置。0.0表示图例左侧与图形的左边界对齐，1.0表示图例右侧与图形的右边界对齐。
# 第二个值（1.0）表示图例的垂直位置。0.0表示图例底部与图形的底部对齐，1.0表示图例顶部与图形的顶部对齐。


# ax = plt.subplot(gs1[0, 2])
ax = fig.add_subplot(gs[1, 2])
ax.plot(x, Exact[index3, :], 'b-', linewidth=1, label='Exact')
#ax.plot(x, U_pred[index3, :], 'r:', linewidth=1, label='Predicted')
ax.plot(x, U_pred[index3, :], 'r:', linewidth=1, label='Predicted')
ax.set_xlabel('$x$')  # x轴标签
ax.set_title('$t = %.2f$(f)' % (t[index3][0]), fontsize=10)
#ax.set_yticks([-7.0, -3.5, 0, 3.5, 7.0])
#ax.set_yticks([ 0, 2, 4, 6])
#ax.set_xticks([ -10, -5, 0, 5, 10])
#ax.set_xticks([ -40, -20, 0, 20, 40])

#ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t+skip], u1.shape[0]), fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(-0.9, -0.25), ncol=2, frameon=False)
#ax.legend(loc='upper center', bbox_to_anchor=(-1.0, -0.25), ncol=2, frameon=False)
# 添加线条注释
#legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 4})
#legend = ax.legend(loc='upper right', frameon=False, prop= fontP)# bbox_to_anchor=(1.0, 1.0), prop={'size': 4})
#legend = ax.legend(loc='upper right',frameon=False, prop= fontP, bbox_to_anchor=(.45, 0.94))# bbox_to_anchor=(1.0, 1.0), prop={'size': 4})

# 文件存储
plt.savefig('./ac-d.pdf', dpi=300, bbox_inches='tight')
#plt.savefig('./ac.svg', dpi=300, bbox_inches='tight')
plt.show()

