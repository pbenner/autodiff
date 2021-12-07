## Copyright (C) 2021 Philipp Benner
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
