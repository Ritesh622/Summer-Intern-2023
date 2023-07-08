import numpy as np

x_norm_sum=0
sum=0
def get_array(d):
    while True:
        arr = np.random.rand(d,)
        norm= np.linalg.norm(arr)
        if norm <= 1:
            return list(arr)
n=10
d=10
x=[]
for _ in range(n):
    x.append(get_array(d))
#print(x)

for j in range(n):
    x_max= max(x[j])
    x_min= min(x[j])
    x_norm=((np.linalg.norm(x[j]))**2)
    x_norm_sum += x_norm
    for i in range(d):
        sum +=(x_max-x[j][i])*(x[j][i]-x_min)
        #print(sum)
    
# print(sum)    
# print(x_norm_sum)   
error=sum/(n**2)
upper_bound=(d*x_norm_sum)/(2*(n**2))
lower_bound=((d-2)*x_norm_sum)/(2*(n**2))
print(f"Error is {error}")
print(f"Upper Bound is {upper_bound}")
print(f"Lower Bound is {lower_bound}")
