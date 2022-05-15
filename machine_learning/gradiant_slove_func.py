import numpy as np
np.random.seed(20)
import matplotlib.pyplot as plt

alpha = 0.1
theta = 1
loop_times = 100

# 函数f
def f(x,y):
    return x**2+y**2

# 对f求偏导，生成梯度
def gradiant(x,y):
    return np.array([2*x,2*y])

# 生成初始值
start_value = np.random.randint(10,100,2)
print(start_value)

ans = [f(start_value[0],start_value[1])]
for _ in range(loop_times):
    grad = gradiant(start_value[0],start_value[1])
    # 更新初始值
    start_value = start_value - alpha*grad
    ans.append(f(start_value[0],start_value[1]))

plt.plot(range(len(ans)),ans)
plt.show()












