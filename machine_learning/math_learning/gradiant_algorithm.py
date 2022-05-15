import numpy as np
import matplotlib.pyplot as plt

"""
构造损失函数，确保为凸函数
如果参数很多，那么利用矩阵进行计算
关键点在于计算出当前函数的梯度
利用梯度下降，随机梯度下降，或者小批量梯度下降迭代求解最优解，对于一个拟合问题来说，就是求出了损失最小的参数w
"""

# 伪造样本
x = np.linspace(0,100,100)
# 考虑到有一个截距
x = np.c_[x,np.ones(100)]
# 方程参数
W = np.array([3,2])
y = x.dot(W)
x = x.astype('float')
y = y.astype('float')
x[:,0] += np.random.normal(size=(x[:,0].shape))*3
y = y.reshape(100,1)
print(y.shape,x.shape,W.shape)

# 线性回归损失函数
def f(x,y,w):
    return (y-x.dot(w)).T.dot(y-x.dot(w)).reshape(-1)

# 线性规划损失函数的梯度
def f_gradiant(x,y,w):
    return -2*x.T.dot(y-x.dot(w))

# 初始化参数
w=np.random.random(size=(2,1))
# print(w)
dw = f_gradiant(x,y,w)

epoches = 1000
eta = 0.000001
ans = [f(x,y,w)]
"""梯度下降，全样本计算"""
# for _ in range(epoches):
#     # 梯度下降
#     dw = f_gradiant(x,y,w)
#     w = w - eta*dw
#     ans.append(f(x,y,w))
"""随机梯度下降，单样本计算"""
# for _ in range(epoches):
#     # 随机梯度下降，只选取1个样本进行梯度下降
#     select_single= np.random.randint(100)
#     tp_y = y[[select_single]]
#     tp_x = x[[select_single]]
#     dw = f_gradiant(tp_x,tp_y,w)
#     w = w - eta*dw
#     ans.append(f(x,y,w))
"""小批量梯度下降"""
for _ in range(epoches):
    # mini_batch
    select_items = np.random.randint(0, 100, 20)
    tp_y = y[[select_items]]
    tp_x = x[[select_items]]
    dw = f_gradiant(tp_x, tp_y, w)
    w = w - eta * dw
    ans.append(f(x, y, w))

plt.plot(range(len(ans)),ans)
plt.show()
print(w,W)
plt.scatter(x[:,0],y)
plt.plot(x[:,0],x.dot(W),'r')
plt.show()















# def f(theta_v)：
#     return 1/2*( - )