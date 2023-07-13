import numpy as np

l = 1 # shared randomness

x = np.random.rand(10)
NUM_ITER = int(1e8)


def encoder(x, h):
     r = np.random.rand() # private randomness
     return (x >= (r+h)*(2**(-l))).astype(np.int8)


def decoder(X, h):
    return X + (h - 0.5*(2**(l)-1))*(2**(-l))

def variance(x):
    a = (1/12) * (1-(4**(-l)))
    b = x % (2**(-l))
    return a + (2**(-l))*b - b**2

max_var = 0
for xi in x:

    h = np.random.randint(low=0, high=2**l, size=NUM_ITER) 
    Xi = encoder(xi, h)
    xi_hat = decoder(Xi, h)
    var = variance(xi)

    print("xi:", xi,
        "\nexpect_xihat:", np.sum(xi_hat)/NUM_ITER)

    print("\nexpectation on h (obs)", np.sum(h)/NUM_ITER)
    print("expectation on h (cal)", (0.5)*(2**(l) - 1))
    print("\nVariance:", var)
    max_var = max(max_var, var)

    print("-------------------------------------------")

print("Maximum variance:", max_var)