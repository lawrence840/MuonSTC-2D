# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:58:19 2022

@author: Muon Group HZY

缪子二维成像算法
"""

import math
import seaborn as sns
import scipy
import sympy
import numpy as np
import scipy.signal as signal 
import matplotlib.pyplot as plt
from pylab import mpl
from scipy.interpolate import interp1d
import random
import os
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

mpl.rcParams["font.sans-serif"] = ["Times New Roman"] #指定默认字体
mpl.rcParams["axes.unicode_minus"] = False #解决保存图像时负号“-”显示为方块的问题

#torch.set_printoptions(profile="full")
np.set_printoptions(threshold = np.inf, precision=3, suppress=True)

# In[1] 输入参数
Em = 0.1  #粒子卡阈能量（单位GeV）
vector_detector = []  #探测器系统法线方向
d = 0.5  #二维密度图每个bin的大小（单位°）
n = int(180 / d + 1)  #二维密度图横纵坐标ticks数
pho = 2.65  #物质的密度（单位g/cm3）
#Phi = 14  #探测器朝向的方位角
#最后成像结果的小矩阵范围(theta and phi 单位°)
thetaMin = 45
thetaMax = 90
phiMin = -25
phiMax = 25
#对应的小矩阵的行列范围为
top = int((180-thetaMin) // d)
bottom = int((180-thetaMax) // d)
left = int((180//2 + phiMin) // d)
right = int((180//2 + phiMax) // d)

# In[2] 读取txt文件并输出二维角度矩阵
# 缓存文件读取
def load_or_cache(filename):
    cache_filename = filename.replace('.txt', '.npy')
    if os.path.exists(cache_filename):
        return np.load(cache_filename)
    data = np.loadtxt(filename)
    np.save(cache_filename, data)
    return data

def readtxt(path):  #读取数据
    vector = []
    data = np.loadtxt(path)
    mask = data[:, 7] >= Em  # 能量筛选
    data = data[mask]

    # 添加高斯分布的不确定度
    error_x = np.random.normal(0, 0.01 * np.sin(np.radians(10)), size=(len(data), 2))
    error_y = np.random.normal(0, 0.01 * np.cos(np.radians(10)), size=(len(data), 2))
    error_z = np.random.normal(0, 0.01, size=(len(data), 2))
    
    # 计算探测器两个hit连线的向量，并存到数组vector中
    point_1 = data[:, [0, 2, 1]] + np.stack([error_x[:, 0], error_y[:, 0], error_z[:, 0]], axis=-1)
    point_2 = data[:, [3, 5, 4]] + np.stack([error_x[:, 1], error_y[:, 1], error_z[:, 1]], axis=-1)
    vector = point_2 - point_1

    return vector 

def angle(vector): #求向量的仰角和方位角
    normal_vector_z = [0, 0, -1] #z轴方向向量
    normal_vector_x = [math.sin(math.radians(190)), math.cos(math.radians(190)), 0] #探测器系统x轴方向向量
    norm2_vector = np.linalg.norm(vector) #求向量的二范数
    project_vector_xoy = [vector[0], vector[1], 0] #向量在xoy平面投影
    norm2_project_vector = np.linalg.norm(project_vector_xoy) #求投影向量的二范数
    if norm2_project_vector != 0:
        theta = math.degrees(math.asin(np.dot(vector, normal_vector_z)/norm2_vector))  #天顶角theta
        phi = math.degrees(math.asin(np.dot(project_vector_xoy, normal_vector_x)/norm2_project_vector)) #方位角phi 
    else:
        theta = 0
        phi = 0
    
    return theta, phi

def Vec2Matrix(heatmapdata, statistics_vector):
    statistics_vector = np.array(statistics_vector)
    x = np.round(statistics_vector[:, 0] / d + 90 // d).astype(int)
    y = np.round(statistics_vector[:, 1] / d + 90 // d).astype(int)
    
    valid_mask = (x >= 0) & (x < heatmapdata.shape[0]) & (y >= 0) & (y < heatmapdata.shape[1])
    x, y = x[valid_mask], y[valid_mask]
    np.add.at(heatmapdata, (x, y), 1)
    return heatmapdata

# In[3]:  求物体厚度角分布（若物体形状不规则或者已在G4中获取则忽略此步骤）

def length(): #读取楼房厚度文件
    heatmapdata = np.loadtxt("length-calculate-36.txt")
    return heatmapdata

# In[4]: 缪子微分通量公式选择与最小能量求解
#积分能量上下限
MINE = 0.1
MAXE = 1000
'''修正Gaisser公式求解'''
from scipy.special import hyp2f1
from scipy.optimize import bisect
def cos_theta_star(cos_theta):  #公式中修正后的cos(theta*)
    p1 = 0.102573
    p2 = 0.068287
    p3 = 0.958633
    p4 = 0.0407253
    p5 = 0.817285
    return ((cos_theta**2 + p1**2 + p2*cos_theta**p3 +p4*cos_theta**p5) / (1 + p1**2 + p2 + p4))**0.5
def F(x, k1, k2):  #公式中右侧括号项的通式
    a = k1 + x
    b = 1 - k1 * k2 
    c = 1 + k2 * x
    return -0.37037 * (k2*a/c)**2.7 * hyp2f1(2.7, 2.7, 3.7, b / c) / (k2 * a ** 2.7)

def Phi(x, cos_theta):  #f(x)积分得到的F(x)，可以用微分超几何函数表示
    q = cos_theta_star(cos_theta)
    k1 = 3.64 / q ** 1.29
    k2 = 1.1 * q / 115
    k3 = 1.1 * q / 850
    return 0.14 * F(x, k1, k2) + 0.00756 * F(x, k1, k3)

def solve_E(R, cos_theta):  #求解最小能量
    def flux(E, cos_theta):  #对于建筑的积分项
        return Phi(MAXE, cos_theta) - Phi(E, cos_theta)
    def total_flux(cos_theta):  #对于开阔天空的积分项
        return Phi(MAXE, cos_theta) - Phi(MINE, cos_theta)
    def f(E):
        return R * total_flux(cos_theta) - flux(E, cos_theta)
    return bisect(f, MINE, MAXE)

# In[6] 新建求ECOMUG积分项

import numpy as np
from scipy import integrate, optimize
import math

# 定义被积函数
def integrand(p, a, n):
    return 1600 * (p + 2.68) ** -3.175 * p ** 0.279 * a ** (n + 1)

def integrand2(p, a, b, n):
    return 1600 * (p + 2.68) ** -3.175 * p ** 0.279 * (math.cos(a)) ** n * math.pi/4 *((math.sin(a))**2 *math.cos(b)+math.cos(a)*math.sin(a))

# 定义 n(p)
def n_function(p):
    return max(0.1, 2.856 - 0.655 * math.log(p))

lower_limit_start = 0.1
upper_limit = 1000.0

def calculate_pixel(i, j, heatmapdata):
    R = heatmapdata[i][j]
    if R > 1:
        R = 1
    result1, _ = integrate.quad(lambda p: integrand2(p, a=(180-i*d)/180*math.pi, b=j*d/180*math.pi, n=n_function(p)), MINE, MAXE)
    target_integral_value = R * result1
    
    def target_function(lower_limit):
        integral_result, _ = integrate.quad(lambda p: integrand2(p, a=(180-i*d)/180*math.pi, b=j*d/180*math.pi, n=n_function(p)), lower_limit, MAXE)
        return integral_result - target_integral_value
    
    return optimize.bisect(target_function, lower_limit_start, upper_limit)

def Emin(heatmapdata):
    heatmapdata0 = np.zeros((n, n))
    with Pool(cpu_count()) as pool:
        results = pool.starmap(calculate_pixel, [(i, j, heatmapdata) for i in range(bottom, top+1) for j in range(left, right+1)])
    for idx, value in enumerate(results):
        i = idx // (right-left+1) + bottom
        j = idx % (right-left+1) + left
        heatmapdata0[i][j] = value
    return heatmapdata0

# In[5]: 最小能量-不透明度公式选择与不透明度求解
method = 5  #重建公式数量

def f(x): #Power公式
    if x != 0:
        a = math.log10(x)
        l0 = 0.2549
        l1 = 0.0801
        l2 = 0.0368
        l3 = -0.0461
        l4 = 0.0154
    else:
        a = l0 = l1 = l2 = l3 = l4 = 0
    return 10 ** ((l4*a**4 + l3*a**3 + l2*a**2 + l1*a + l0)) / 1000

def g(x):  #Quad公式
    if x != 0:
        s = math.log10(x)
        a = 1.8825 + 0.2885*s - 0.0175*s**2
        b = 0.245 + 1.861*s - 0.205*s**2
        return (a+b*x/1000)/1000
    else:  return 0

data_power = np.loadtxt('output_power.txt')
x_values_p = data_power[:, 0]
y_values_p = data_power[:, 1]

# 创建插值函数
interp_function_p = interp1d(y_values_p, x_values_p, kind='linear', fill_value='extrapolate')

data_quad= np.loadtxt('output_quad.txt')
x_values_q = data_quad[:, 0]
y_values_q = data_quad[:, 1]

# 创建插值函数
interp_function_q = interp1d(y_values_q, x_values_q, kind='linear', fill_value='extrapolate')


def Opacity(heatmapdata):
    heatmapdata_array = np.zeros((method, n, n))  # 创建5个n*n的heatmap的数组储存不同方法的结果

    # 用来存储每个(i, j)位置的计算任务
    tasks = []

    def compute(i, j):
        stop = heatmapdata[i][j]
        result = np.zeros(5)
        
        # 常数法
        result[0] = heatmapdata[i][j] * 1000 / 2.5
        
        # 二次拟合法
        result[1] = interp_function_q(stop)
        
        # 对数拟合法
        result[2] = float(25000 * (math.log(0.021164 * heatmapdata[i][j] + 1))) 
        
        # 三次多项式法
        x = sympy.Symbol('x')
        result[3] = sympy.solve(2.62 * x + 165.98 * 10**(-13) * x**3 - 1000 * stop)[0]
        
        # 能损积分法
        result[4] = interp_function_p(stop)
        
        return (i, j, result)

    # 使用ThreadPoolExecutor并行计算
    with ThreadPoolExecutor() as executor:
        for i in range(int(bottom), int(top) + 1):
            for j in range(int(left), int(right) + 1):
                tasks.append(executor.submit(compute, i, j))
        
        # 等待并获取结果
        for task in tasks:
            i, j, result = task.result()
            heatmapdata_array[:, i, j] = result

    return heatmapdata_array


# In[5]:滤波与边缘去噪
k = 3  #滤波器的大小
sigma = 0.1  #高斯滤波器宽度

def MidFilter(heatmapdata, k):  #中值滤波
    heatmapdata1 = signal.medfilt(heatmapdata,(k,k))
    return heatmapdata1

def UpThreshold(heatmapdata, maximum):  #卡掉大于上限阈值的像素
    heatmapdata1 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):               
            if heatmapdata[i][j] > maximum:
                heatmapdata1[i][j] = maximum
            else: heatmapdata1[i][j] = heatmapdata[i][j]
    #np.clip(heatmapdata, 0, maximum, out = heatmapdata)
    return heatmapdata1

def DownThreshold(heatmapdata, minimum):  #卡掉小于下限阈值的像素并赋值0
    heatmapdata1 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):               
            if heatmapdata[i][j] < minimum:
                heatmapdata1[i][j] = 0
            else: heatmapdata1[i][j] = heatmapdata[i][j]
    return heatmapdata1

def Outside(heatmapdata1,heatmapdata2):  #将热力图2中的热力图1零值区域的部分取出
    heatmapdata = np.zeros((n, n))
    for i in range(n):
        for j in range(n):               
            if heatmapdata1[i][j] != 0:
                heatmapdata[i][j] = 0
            else: heatmapdata[i][j] = heatmapdata2[i][j]
    return heatmapdata

def Inside(heatmapdata1,heatmapdata2):  #将热力图2中的热力图1非零值区域的部分取出
    heatmapdata = np.zeros((n, n))
    for i in range(n):
        for j in range(n):               
            if heatmapdata1[i][j] == 0:
                heatmapdata[i][j] = 0
            else: heatmapdata[i][j] = heatmapdata2[i][j]
    return heatmapdata

def Nonzero(heatmapdata):  #提取热力图中非零元素
    data = heatmapdata.flatten()[np.flatnonzero(heatmapdata)]
    return data

def Normalization(heatmapdata1, heatmapdata2):  #将热力图1中一定区域的值按热力图2归一化
    total1 = np.sum(heatmapdata1)
    total2 = np.sum(heatmapdata2)
    heatmapdata0 = np.zeros((n, n))
    if total1 != 0:
        for i in range(n):
            for j in range(n):
                heatmapdata0[i][j] = heatmapdata1[i][j] * total2 / total1
    return heatmapdata0

# In[6]: 绘图
def compute_ratio(heatmapdata1,heatmapdata2,ratio):  #求两个热力图比值
    heatmapdata = np.zeros((n, n))
    ratio = ratio*1.0
    for i in range(n):
        for j in range(n):
            if heatmapdata2[i][j] != 0:
                heatmapdata[i][j] = float(heatmapdata1[i][j]/heatmapdata2[i][j]/ratio)
            else:
                heatmapdata[i][j] = 0
    return heatmapdata

def plt_Density_Profile(heatmapdata, path, ratio, minus):  
    """
    主绘图函数，绘制热力图并保存为PNG格式。
    
    参数：
    heatmapdata -- 热力图数据（二维数组）
    path -- 文件路径，用于保存热力图
    ratio -- 用于确定colorbar上限的比例因子
    minus -- 用于确定colorbar下限的偏移量
    """
    # 确保path合法，并替换无效字符
    savename = os.path.splitext(path)[0] + '.png'
    print(f"Saving figure to: {savename}")
    
    titlename = os.path.splitext(path)[0]

    # 设置画布大小
    f, ax = plt.subplots(figsize=(10, 6))

    # 定义坐标轴刻度
    xticklabels = [m_x*d if (m_x*d) % 10 == 0 else ' ' for m_x in range(int(phiMin//d), int(phiMax//d) + 1)]
    yticklabels = [90 - m_y*d if (m_y*d) % 10 == 0 else ' ' for m_y in range(0, 91)]

    # 取出矩阵的一部分进行绘图
    heatmapdata = heatmapdata[bottom-1:top, left-1:right]
    
    # 保存部分矩阵到txt文件
    np.savetxt(os.path.splitext(path)[0] + 'out.txt', heatmapdata, fmt='%f', delimiter=' ')
    
    # 判断热力图数据是否为空
    if heatmapdata.size == 0:
        print("Array was empty.")
        return

    # 设置colorbar的最大最小值
    vmax = heatmapdata.max() * ratio
    vmin = heatmapdata.min() - vmax * minus
    print(f"Colorbar limits: vmin={vmin}, vmax={vmax}")
    
    # 掩盖零值区域
    mask = heatmapdata <= 0
    
    # 绘制热力图
    heatmap = sns.heatmap(heatmapdata, fmt='g', vmin=vmin, vmax=vmax, center=(vmin + vmax) / 2,
                          xticklabels=xticklabels, yticklabels=yticklabels,
                          cmap='jet', mask=mask, square=True, cbar=False)
    
    # 添加颜色条并设置字号
    cb = heatmap.figure.colorbar(heatmap.collections[0])
    cb.ax.tick_params(labelsize=20)
    
    # 设置坐标轴标签字体大小
    plt.setp(ax.get_yticklabels(), rotation=360, fontsize=20)
    plt.setp(ax.get_xticklabels(), rotation=360, fontsize=20)
    ax.invert_yaxis()
    ax.set_xlabel('Phi (deg)', fontsize=20)
    ax.set_ylabel('Theta (deg)', fontsize=20)
    
    # 设置图片边框样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
    
    # 保存图片并确保不会出现保存失败的情况
    try:
        plt.savefig(savename, dpi=1600, bbox_inches='tight')
        print(f"Figure saved successfully as {savename}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    
    # 展示图像
    #plt.show()
    
    # 关闭绘图对象，避免内存泄漏或未关闭图形
    plt.close(f)

def Density_Profile(statistics_vector,path):
    # 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
    heatmapdata = np.zeros((n,n))
    heatmapdata = Vec2Matrix(heatmapdata,statistics_vector)
    heatmapdata = MidFilter(heatmapdata, k)
    plt_Density_Profile(heatmapdata, path, 1, 0)
    return heatmapdata

# In[7]: 主程序与文件输出
'''主程序'''
def main(path):
    ##读取数据
    vector = readtxt(path)
    ##求其天顶角与方位角
    statistics_vector = []
    for i,m in enumerate(vector):
        theta, phi = angle(m)
        statistics_vector += [[theta, phi]]
    ##作二维密度分布图
    heatmapdata = Density_Profile(statistics_vector,path)
    return heatmapdata

if __name__ == '__main__':
    
    path1 = 'BF2.txt'
    print('计算文件1二维密度分布图')
    heatmapdata1=main(path1)  #文件1二维密度分布图
    path2 = 'sky2.txt'
    print('计算文件2二维密度分布图')
    heatmapdata2=main(path2)  #文件2二维密度分布图

    print('计算文件2与1比值得二维密度分布图')  # 文件1/2比值二维密度分布图
    heatmapdata3 = compute_ratio(heatmapdata1,heatmapdata2,1/1)
    path3=path1.split('.')[0]+'-'+path2.split('.')[0]+' '+'ratio.abc'
    print(path3)
    heatmapdata3 = UpThreshold(heatmapdata3, 2)
    plt_Density_Profile(heatmapdata3, path3,1,0)
    
    print('计算文件山体密度分布图')  #山体密度图
    heatmapdata4=length()
    path4 = 'length of the building'
    plt_Density_Profile(heatmapdata4, path4,1,0)

    print('输出山体密度分布图文件')  #最终密度图
    heatmapdata5_array = np.zeros((method, n, n))
    
    path5 = ['Const-g', 'Quad-g', 'Log-g', 'Poly-g', 'Intg-g']
    path8 = ['Const-O', 'Quad-O', 'Log-O', 'Poly-O', 'Intg-O']
    
    temp1 = Emin(heatmapdata3)
    print('最小能量分布')  #最小能量
    heatmapdata13 = temp1
    path13 = 'Emin'
    plt_Density_Profile(heatmapdata13, path13,0.1,0)
    temp3 = Inside(heatmapdata4, temp1)
    temp2 = Opacity(temp1)
    
    for s in range(method):
        temp2[s] = UpThreshold(temp2[s], 5000)
        plt_Density_Profile(temp2[s], path8[s],1, 0)
        heatmapdata5_array[s] = compute_ratio(temp2[s],heatmapdata4,100)
        heatmapdata5_array[s] = UpThreshold(heatmapdata5_array[s], 10)
        plt_Density_Profile(heatmapdata5_array[s], path5[s],1, 0)