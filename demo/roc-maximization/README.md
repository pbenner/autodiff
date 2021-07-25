
Maximize the ROC curve of quadratic discriminant analysis. The objective function is:

<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/roc-maximization/README//eq_no_01.png" alt="" height="60">

where 
<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/roc-maximization/README//eq_no_02.png" alt="" height="60">
 denotes the true positive rate and FPR the false positive rate of the classifier. The parameters of the quadratic discriminant analysis is denoted by 
<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/roc-maximization/README//eq_no_03.png" alt="" height="60">
 and includes the means and covariance matrices of both normal distributions. Gradient ascent is used to maximize 
<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/roc-maximization/README//eq_no_04.png" alt="" height="60">
 with respect to 
<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/roc-maximization/README//eq_no_05.png" alt="" height="60">
.

![Optimization](roc.plots/plot.gif)
