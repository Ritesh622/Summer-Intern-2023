import numpy as np
#l=int(input("Enter number of bits of shared randomness:\n"))
cost=[]
x= np.random.rand()
for j in range(1000):
    h= np.random.randint(0,2)
    #x1.append(x)
    #print(h)
    # print(x)
    if x>=(0.4 + 0.2*h):
        X=1
    else:
        X=0
    # print(X)
    x_estimated= (0.1 + 0.2*h + 0.6*X)
    #estimate.append(x_estimated)

    error=abs((x_estimated-x)**2)
    cost.append(error)

max_cost=max(cost)

print("Worst Case cost is {}".format(max_cost))

#Hence proved
#print(h)