import numpy as np
np.random.seed(20)
import collections

# make a random list
a = np.random.randint(0,9,100)
# calcu the distribute
a_distribute =collections.Counter(a)
for key,v in a_distribute.items():
    a_distribute[key]/= 100
def calcu_cross_entropy(p,q):
    ans = 0
    for k,v in p.items():
        ans += p[k]*np.log(1/q[k])
    return ans
print(calcu_cross_entropy(a_distribute,a_distribute))