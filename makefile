
SUBDIRS = \
	. \
	algorithm/backSubstitution \
	algorithm/bfgs \
	algorithm/blahut \
	algorithm/cholesky \
	algorithm/determinant \
	algorithm/eigensystem \
	algorithm/gaussJordan \
	algorithm/gradientDescent \
	algorithm/gramSchmidt \
	algorithm/hessenbergReduction \
	algorithm/householderBidiagonalization \
	algorithm/lineSearch \
	algorithm/matrixInverse \
	algorithm/msqrt \
	algorithm/msqrtInv \
	algorithm/newton \
	algorithm/svd \
	algorithm/qrAlgorithm \
	algorithm/rprop \
	demo/channel \
	demo/entropy \
	demo/example1 \
	demo/regression \
	demo/rosenbrock \
	simple \
	special \
	statistics \
	statistics/generic \
	statistics/matrixDistribution \
	statistics/matrixEstimator \
	statistics/scalarDistribution \
	statistics/scalarEstimator \
	statistics/vectorDistribution \
	statistics/vectorEstimator \
	statistics/matrixClassifier \
	statistics/vectorClassifier \
	statistics/scalarClassifier \

all:

test:
	@for i in $(SUBDIRS); do \
		echo "Testing $$i"; (cd $$i && go test); \
	done
