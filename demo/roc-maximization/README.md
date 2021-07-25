
Maximize the ROC curve of quadratic discriminant analysis. The objective function is:

<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/roc-maximization/README//eq_no_01.png" alt="" height="60">

where TPR denotes the true positive rate and FPR the false positive rate of the classifier, both as a function of the threshold t. The parameters of the quadratic discriminant analysis given by the vector 
<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/roc-maximization/README//eq_no_02.png" alt="" height="60">
, which includes the means and covariance matrices of both normal distributions. Gradient ascent is used to maximize f with respect to x.

![Optimization](roc.plots/plot.gif)
