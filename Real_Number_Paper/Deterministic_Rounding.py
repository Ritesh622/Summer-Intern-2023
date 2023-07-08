import numpy as np
x1=[]
cost=[]
estimate=[]

for j in range(5000):
    x= np.random.rand()
    #x=0.5
    x1.append(x)

    if x>=0.5:
        X=1
    else:
        X=0

    x_estimated= X/2 + 0.25
    estimate.append(x_estimated)
    error=abs((x_estimated-x)**2)
    cost.append(error)
    max_cost=max(cost)
    for index,cost1 in enumerate(cost):
        if cost1==max_cost:
            i=index
            break

print(f"Max cost of {max_cost} occurs at x={x1[i]} and X={estimate[i]}")
#Verified that max cost is 1/16 and occurs at 0,0.5 or 1(but why?)