import numpy as np
l=int(input("Enter number of bits of shared randomness:\n"))
var=[]
for j in range(1000):
    h= np.random.randint(0,2**l-1)
    r=np.random.rand()
    x=np.random.rand()
    #x1.append(x)
    if x>=(r+h)*(2**(-l)):
        X=1
    else:
        X=0

    x_estimated=X+(h-0.5*(2**l-1))*(2**(-l))
    #estimate.append(x_estimated)

    variance=(1/12)*(1-(4**(-l))) + (2**(-l))*x - x**2
    var.append(variance)

max_var=max(var)
print("Worst Case Variance is {}".format(max_var))

#Hence proved