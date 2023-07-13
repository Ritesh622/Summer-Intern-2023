from scipy.stats import bernoulli
import numpy as np


def encoder(x, num_outs=1):
    ber = bernoulli(x)
    return ber.rvs(num_outs)


def decoder(X):
    return X


x = np.random.rand(1000)

# for single example
# print("x:", x)

# X = encoder(x)
# print('X:', X)

# x_hat = decoder(X)
# print('x_hat:', x_hat)

NUM_ITER = int(1e2)
# for xi in x:
#     sum_xi_hat = 0
#     for iter in range(NUM_ITER):
#         Xi = encoder(xi)
#         xi_hat = decoder(Xi)
#         sum_xi_hat += xi_hat
#     print(
#         f"x: {xi}; \
#         expectation: {sum_xi_hat/NUM_ITER} \
#             variance: {xi * (1-xi)}"
#     )
worst_case_var = 0
worst_case_var_at = 0
for xi in x:
    Xi_encs = encoder(xi, num_outs=NUM_ITER)
    xi_hat_decs = decoder(Xi_encs)

    expect_xi = np.sum(xi_hat_decs) / NUM_ITER
    variance = xi * (1 - xi)

    if variance > worst_case_var:
        worst_case_var = variance
        worst_case_var_at = xi

    print(
        f"xi: {xi}\
        expectation: {expect_xi}\
        variance: {variance}"
    )
print("\nworst case variance:", worst_case_var,
      "at:", worst_case_var_at)
