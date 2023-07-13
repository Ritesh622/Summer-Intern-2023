import numpy as np

def encoder(x):
    return (x>=0.5).astype(np.int8)

def decoder(X):
    return (0.5 * X) + 0.25

x = np.random.rand(1000)

X = encoder(x)
x_hat = decoder(X)

abs_loss = abs(x-x_hat)
se_loss = (x - x_hat)**2

print(
    f"x: {x}\n",
    f"X: {X}\n"
    f"x_hat: {x_hat}\n",
    f"abs_loss: {abs_loss}\n",
    f"se_loss: {se_loss}\n"
)

print("max_abs_loss", max(abs_loss))
print("max_se_loss", max(se_loss))

print("\nloss is maximum at:")
print(x[(abs(se_loss-max(se_loss)) <= 1e-3)])