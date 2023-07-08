import numpy as np
x1=[]
x_estimated=[]
var=[]
for j in range (1000):
    x= np.random.rand()
    #print(x.item())
    x1.append(x)

    #Bernoulli Distribution
    random_value=np.random.rand()
    #print(random_value.item())
    if random_value<=x:
        X=1
    else:
        X=0
    #print(X)
    x_estimated.append(X)
    variance=x*(1-x)
    var.append(variance)
max_var=max(var)
for index,var1 in enumerate(var):
    if var1==max_var:
        i=index
        break
print(f"Max Variance of {max_var} occurs at x={x1[i]} and X={x_estimated[i]}")
