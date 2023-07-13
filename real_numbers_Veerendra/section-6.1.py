import numpy as np

NUM_ITER = int(1e5)

x = (0, 0.5, 1)
h = np.random.randint(0, 2, size=NUM_ITER)
# alpha = np.random.rand()
alpha = (1-(1/np.sqrt(2)))

def encoder(x):
    # if (x==1) or ((x==0.5) and (h==0)):
    #     return 1
    # return 0
    return np.bitwise_or((x==1), np.bitwise_and((x==0.5), (h==0)))


def decoder(X):
    return alpha * h + (1-alpha) * X

# print("h:", h)
print("alpha:", alpha)
print("alpha squared", (alpha**2)/2)
print("0.5alpha", (alpha-0.5)**2)
print("-----------------\n")
for xi in x:
    print("xi:", xi)
    Xi = encoder(xi)
    xi_hat = decoder(xi)
    error = (xi - xi_hat)**2
    # print("error:", error)
    print("min error:", np.min(error))
    print("max error:", np.max(error))

    print("expected_value of error", sum(error)/NUM_ITER)
    print("------------------------")
