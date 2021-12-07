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

import numpy as np

from sklearn.datasets import make_regression

## -----------------------------------------------------------------------------

# number of data points
n = 100
# number of features
p = 100

X, y_ = make_regression(n_samples=n, n_features=p)

y = np.zeros(n)
y[y_ > np.median(y_)] = 1

np.savetxt("logisticData.X.table", X, delimiter=" ")
np.savetxt("logisticData.y.table", y, delimiter=" ")
