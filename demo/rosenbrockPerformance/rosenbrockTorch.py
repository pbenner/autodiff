
import time
import torch as tr

## objective function
## -----------------------------------------------------------------------------

def rosenbrock(z):
    a =   1.0
    b = 100.0
    x = z[0]
    y = z[1]
    return tr.pow((a - x), 2.0) + b*tr.pow(y - x*x, 2.0)

## -----------------------------------------------------------------------------

def measure_time(closure):
    t0 = time.time()
    closure()
    t1 = time.time()
    return t1-t0

## -----------------------------------------------------------------------------

# number of steps
T = 10000

def test():
    # define initial value
    x = tr.tensor([-10.0,10.0], requires_grad=True)
    optimizer = tr.optim.Adam([x])
    # make T optimization steps
    for i in range(1, T):
        optimizer.zero_grad()
        y = rosenbrock(x)
        y.backward()
        optimizer.step()
    print(x)

## -----------------------------------------------------------------------------

elapsed=measure_time(test)

print(f'{elapsed}s')
