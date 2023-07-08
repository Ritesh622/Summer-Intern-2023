import numpy as np
import random

# x=bernoulli(0.5)
# y=bernoulli.rvs(0.8,size=10)
# print(y)
n=2
d=10
x_est_mean=0
y=[]
for i in range(n):
    x=np.random.rand(d,)
    #print(x)
    x=list(x)
    x_max=max(x)    
    x_min=min(x)
    for j in range(d):
        p=(x[j]-x_min)/(x_max-x_min)
        x_est=random.choices([x_max,x_min],[p,1-p])[0]
        y.append(x_est)
        x_est_mean+=x_est
    x_est_mean=(x_est)/n
    print("\n",x_est_mean)
    print(y,"\n")