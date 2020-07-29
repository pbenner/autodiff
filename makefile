
SUBDIRS = \
	. \
	algorithm/backSubstitution \
	algorithm/bfgs \
	algorithm/blahut \
	algorithm/blahut/blahut-R \
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
	algorithm/saga \
	algorithm/svd \
	algorithm/qrAlgorithm \
	algorithm/rprop \
	demo/channel \
	demo/entropy \
	demo/example1 \
	demo/regression \
	demo/rosenbrock \
	special \
	statistics \
	statistics/generic \
	statistics/matrixClassifier \
	statistics/matrixDistribution \
	statistics/matrixEstimator \
	statistics/scalarClassifier \
	statistics/scalarDistribution \
	statistics/scalarEstimator \
	statistics/vectorClassifier \
	statistics/vectorDistribution \
	statistics/vectorEstimator

all:

test:
	@for i in $(SUBDIRS); do \
		echo "Testing $$i"; (cd $$i && go test); \
	done
