
Find the capacity of a channel between *X* and *Y* by maximizing the mutual information

<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/channel/README//eq_no_01.png" alt="" height="60">


The gradient based approache uses Newton's method to find a critical point of the Lagrangian

<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/channel/README//eq_no_02.png" alt="" height="60">

where probabilities are represented on log-scale.

![Optimization](channel.png)
