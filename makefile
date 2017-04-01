
SUBDIRS = \
	. \
	algorithm/bfgs \
	algorithm/blahut \
	algorithm/cholesky \
	algorithm/determinant \
	algorithm/gaussJordan \
	algorithm/gradientDescent \
	algorithm/gramSchmidt \
	algorithm/hessenbergReduction \
	algorithm/lineSearch \
	algorithm/matrixInverse \
	algorithm/msqrt \
	algorithm/msqrtInv \
	algorithm/newton \
	algorithm/qrAlgorithm \
	algorithm/rprop \
	demo/channel \
	demo/entropy \
	demo/example1 \
	demo/regression \
	demo/rosenbrock \
	distribution \
	special

all:

test:
	@for i in $(SUBDIRS); do \
		echo "Testing $$i"; (cd $$i && go test); \
	done
