
import numpy as np
import time
import torch as tr

## objective function
## -----------------------------------------------------------------------------

class Net(tr.nn.Module):
    def __init__(self, m):
        super(Net, self).__init__()
        self.linear1 = tr.nn.Linear(m, 1, bias=False)
    def forward(self, x):
        x = self.linear1(x)
        x = tr.sigmoid(x)
        return x

## -----------------------------------------------------------------------------

def measure_time(closure):
    t0 = time.time()
    closure()
    t1 = time.time()
    return t1-t0

## -----------------------------------------------------------------------------

# import data
X = tr.from_numpy(np.genfromtxt('logisticData.X.table', delimiter=' ')).float()
y = tr.from_numpy(np.genfromtxt('logisticData.y.table', delimiter=' ')).float()

# number of steps
T = 1000
m = 100

loss_function = tr. nn.BCELoss()
net = Net(m)
net.linear1.weight.data.fill_(0.0)
# define initial value
optimizer = tr.optim.Adam(net.parameters())

def test():
    # make T optimization steps
    for i in range(1, T):
        optimizer.zero_grad()
        y_hat = tr.flatten(net(X))
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()

## -----------------------------------------------------------------------------

elapsed=measure_time(test)

print(list(net.parameters()))
print(f'{elapsed}s')
