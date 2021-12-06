## Documentation

Autodiff is a numerical optimization and linear algebra library for the Go / Golang programming language. It implements basic automatic differentation for many mathematical routines. The documentation of this package can be found [here](https://godoc.org/github.com/pbenner/autodiff).

## Scalars

Autodiff defines three different scalar types. A *Scalar* contains a single mutable value that can be the result of a mathematical operation, whereas the value of a *ConstScalar* is constant and fixed when the scalar is created. Automatic differentiation is implemented by *MagicScalar* types that allow to compute first and second order derivatives. Autodiff supports the following scalars types:

| Scalar       | Implemented interfaces
|--------------|------------------------------------------------------ |
| ConstInt8    | ConstScalar                                           |
| ConstInt16   | ConstScalar                                           |
| ConstInt32   | ConstScalar                                           |
| ConstInt64   | ConstScalar                                           |
| ConstInt     | ConstScalar                                           |
| ConstFloat32 | ConstScalar                                           |
| ConstFloat64 | ConstScalar                                           |
| Int8         | ConstScalar, Scalar                                   |
| Int16        | ConstScalar, Scalar                                   |
| Int32        | ConstScalar, Scalar                                   |
| Int64        | ConstScalar, Scalar                                   |
| Int          | ConstScalar, Scalar                                   |
| Float32      | ConstScalar, Scalar                                   |
| Float64      | ConstScalar, Scalar                                   |
| Real32       | ConstScalar, Scalar, MagicScalar                      |
| Real64       | ConstScalar, Scalar, MagicScalar                      |

The *ConstScalar*, *Scalar* and *MagicScalar* interfaces define the following operations:

| Function     | Description                                           |
|--------------|------------------------------------------------------ |
| GetInt8      | Get value as int8                                     |
| GetInt16     | Get value as int16                                    |
| GetInt32     | Get value as int32                                    |
| GetInt64     | Get value as int64                                    |
| GetInt       | Get value as int                                      |
| GetFloat32   | Get value as float32                                  |
| GetFloat64   | Get value as float64                                  |
| Equals       | Check if two constants are equal                      |
| Greater      | True if first constant is greater                     |
| Smaller      | True if first constant is smaller                     |
| Sign         | Returns the sign of the scalar                        |

The *Scalar* and *MagicScalar* interfaces define the following operations:

| Function     | Description                                           |
|--------------|------------------------------------------------------ |
| SetInt8      | Set value by passing an int8 variable                 |
| SetInt16     | Set value by passing an int16 variable                |
| SetInt32     | Set value by passing an int32 variable                |
| SetInt64     | Set value by passing an int64 variable                |
| SetInt       | Set value by passing an int variable                  |
| SetFloat32   | Set value by passing an float32 variable              |
| SetFloat64   | Set value by passing an float64 variable              |

The *Scalar* and *MagicScalar* interfaces define the following mathematical operations:

| Function     | Description                                           |
| ------------ | ----------------------------------------------------- |
| Min          | Minimum                                               |
| Max          | Maximum                                               |
| Abs          | Absolute value                                        |
| Sign         | Sign                                                  |
| Neg          | Negation                                              |
| Add          | Addition                                              |
| Sub          | Substraction                                          |
| Mul          | Multiplication                                        |
| Div          | Division                                              |
| Pow          | Power                                                 |
| Sqrt         | Square root                                           |
| Exp          | Exponential function                                  |
| Log          | Logarithm                                             |
| Log1p        | Logarithm of 1+x                                      |
| Log1pExp     | Logarithm of 1+Exp(x)                                 |
| Logistic     | Standard logistic function                            |
| Erf          | Error function                                        |
| Erfc         | Complementary error function                          |
| LogErfc      | Log complementary error function                      |
| Sigmoid      | Numerically stable sigmoid function                   |
| Sin          | Sine                                                  |
| Sinh         | Hyperbolic sine                                       |
| Cos          | Cosine                                                |
| Cosh         | Hyperbolic cosine                                     |
| Tan          | Tangent                                               |
| Tanh         | Hyperbolic tangent                                    |
| LogAdd       | Addition on log scale                                 |
| LogSub       | Substraction on log scale                             |
| SmoothMax    | Differentiable maximum                                |
| LogSmoothMax | Differentiable maximum on log scale                   |
| Gamma        | Gamma function                                        |
| Lgamma       | Log gamma function                                    |
| Mlgamma      | Multivariate log gamma function                       |
| GammaP       | Lower incomplete gamma function                       |
| BesselI      | Modified Bessel function of the first kind            |
| LogBesselI   | Log of the Modified Bessel function of the first kind |

## Vectors and Matrices

Autodiff implements dense and sparse vectors and matrices that support basic linear algebra operations. The following vector and matrix types are provided by autodiff:

| Type                     | Scalar       | Description                            |
|--------------------------|--------------|----------------------------------------|
| DenseInt8Vector          | Int8         | Dense vector of Int8 scalars           |
| DenseInt16Vector         | Int16        | Dense vector of Int16 scalars          |
| DenseInt32Vector         | Int32        | Dense vector of Int32 scalars          |
| DenseInt64Vector         | Int64        | Dense vector of Int64 scalars          |
| DenseIntVector           | Int          | Dense vector of Int scalars            |
| DenseFloat32Vector       | Float32      | Dense vector of Float32 scalars        |
| DenseFloat64Vector       | Float64      | Dense vector of Float64 scalars        |
| DenseReal32Vector        | Real32       | Dense vector of Real32 scalars         |
| DenseReal64Vector        | Real64       | Dense vector of Real64 scalars         |
| SparseInt8Vector         | Int8         | Sparse vector of Int8 scalars          |
| SparseInt16Vector        | Int16        | Sparse vector of Int16 scalars         |
| SparseInt32Vector        | Int32        | Sparse vector of Int32 scalars         |
| SparseInt64Vector        | Int64        | Sparse vector of Int64 scalars         |
| SparseIntVector          | Int          | Sparse vector of Int scalars           |
| SparseFloat32Vector      | Float32      | Sparse vector of Float32 scalars       |
| SparseFloat64Vector      | Float64      | Sparse vector of Float64 scalars       |
| SparseReal32Vector       | Real32       | Sparse vector of Real32 scalars        |
| SparseReal64Vector       | Real64       | Sparse vector of Real64 scalars        |
| SparseConstInt8Vector    | ConstInt8    | Sparse vector of ConstInt8 scalars     |
| SparseConstInt16Vector   | ConstInt16   | Sparse vector of ConstInt16 scalars    |
| SparseConstInt32Vector   | ConstInt32   | Sparse vector of ConstInt32 scalars    |
| SparseConstInt64Vector   | ConstInt64   | Sparse vector of ConstInt64 scalars    |
| SparseConstIntVector     | ConstInt     | Sparse vector of ConstInt scalars      |
| SparseConstFloat32Vector | ConstFloat32 | Sparse vector of ConstFloat32 scalars  |
| SparseConstFloat64Vector | ConstFloat64 | Sparse vector of ConstFloat64 scalars  |
| DenseInt8Matrix          | Int8         | Dense matrix of Int8 scalars           |
| DenseInt16Matrix         | Int16        | Dense matrix of Int16 scalars          |
| DenseInt32Matrix         | Int32        | Dense matrix of Int32 scalars          |
| DenseInt64Matrix         | Int64        | Dense matrix of Int64 scalars          |
| DenseIntMatrix           | Int          | Dense matrix of Int scalars            |
| DenseFloat32Matrix       | Float32      | Dense matrix of Float32 scalars        |
| DenseFloat64Matrix       | Float64      | Dense matrix of Float64 scalars        |
| DenseReal32Matrix        | Real32       | Dense matrix of Real32 scalars         |
| DenseReal64Matrix        | Real64       | Dense matrix of Real64 scalars         |
| SparseInt8Matrix         | Int8         | Sparse matrix of Int8 scalars          |
| SparseInt16Matrix        | Int16        | Sparse matrix of Int16 scalars         |
| SparseInt32Matrix        | Int32        | Sparse matrix of Int32 scalars         |
| SparseInt64Matrix        | Int64        | Sparse matrix of Int64 scalars         |
| SparseIntMatrix          | Int          | Sparse matrix of Int scalars           |
| SparseFloat32Matrix      | Float32      | Sparse matrix of Float32 scalars       |
| SparseFloat64Matrix      | Float64      | Sparse matrix of Float64 scalars       |
| SparseReal32Matrix       | Real32       | Sparse matrix of Real32 scalars        |
| SparseReal64Matrix       | Real64       | Sparse matrix of Real64 scalars        |

Autodiff defines three vector interfaces *ConstVector*, *Vector*, and *MagicVector*:

| Interface                        | Function          | Description                                               |
|----------------------------------|-------------------|-----------------------------------------------------------|
| ConstVector                      | Dim               | Return the length of the vector                           |
| ConstVector                      | Equals            | Returns true if the two vectors are equal                 |
| ConstVector                      | Table             | Converts vector to a string                               |
| ConstVector                      | Int8At            | Returns the scalar at the given position as int8          |
| ConstVector                      | Int16At           | Returns the scalar at the given position as int16         |
| ConstVector                      | Int32At           | Returns the scalar at the given position as int32         |
| ConstVector                      | Int64At           | Returns the scalar at the given position as int64         |
| ConstVector                      | IntAt             | Returns the scalar at the given position as int           |
| ConstVector                      | Float32At         | Returns the scalar at the given position as Float32       |
| ConstVector                      | Float64At         | Returns the scalar at the given position as Float64       |
| ConstVector                      | ConstAt           | Returns the scalar at the given position as *ConstScalar* |
| ConstVector                      | ConstSlice        | Returns a slice as a constant vector (*ConstVector*)      |
| ConstVector                      | AsConstMatrix     | Convert vector to a matrix of type *ConstMatrix*          |
| ConstVector                      | ConstIterator     | Returns a constant iterator                               |
| ConstVector                      | CloneConstVector  | Return a deep copy of the vector as *ConstVector*         |
| ConstVector, Vector              | At                | Return the scalar the given index                         |
| ConstVector, Vector              | Reset             | Set all scalars to zero                                   |
| ConstVector, Vector              | Set               | Set the value and derivatives of a scalar                 |
| ConstVector, Vector              | Slice             | Return a slice of the vector                              |
| ConstVector, Vector              | Export            | Export vector to file                                     |
| ConstVector, Vector              | Permute           | Permute elements of the vector                            |
| ConstVector, Vector              | ReverseOrder      | Reverse the order of vector elements                      |
| ConstVector, Vector              | Sort              | Sort vector elements                                      |
| ConstVector, Vector              | AppendScalar      | Append a single scalar to the vector                      |
| ConstVector, Vector              | AppendVector      | Append another vector                                     |
| ConstVector, Vector              | Swap              | Swap two elements of the vector                           |
| ConstVector, Vector              | AsMatrix          | Convert vector to a matrix                                |
| ConstVector, Vector              | Iterator          | Returns an iterator                                       |
| ConstVector, Vector              | CloneVector       | Return a deep copy of the vector as *Vector*              |
| ConstVector, Vector, MagicVector | MagicAt           | Returns the scalar at the given position as *MagicScalar* |
| ConstVector, Vector, MagicVector | MagicSlice        | Resutns a slice as a magic vector (*MagicVector*)         |
| ConstVector, Vector, MagicVector | AppendMagicScalar | Append a single magic scalar                              |
| ConstVector, Vector, MagicVector | AppendMagicVector | Append a magic vector                                     |
| ConstVector, Vector, MagicVector | AsMagicMatrix     | Convert vector to a matrix of type *MagicMatrix*          |
| ConstVector, Vector, MagicVector | CloneMagicVector  | Return a deep copy of the vector as *MagicVector*         |

Vectors support the following mathematical operations:

| Function | Description                      |
| -------- | -------------------------------- |
| VaddV    | Element-wise addition            |
| VsubV    | Element-wise substraction        |
| VmulV    | Element-wise multiplication      |
| VdivV    | Element-wise division            |
| VaddS    | Addition of a scalar             |
| VsubS    | Substraction of a scalar         |
| VmulS    | Multiplication with a scalar     |
| VdivS    | Division by a scalar             |
| VdotV    | Dot product                      |

Autodiff defines three matrix interfaces *ConstMatrix*, *Matrix*, and *MagicMatrix*:

| Interface                        | Function          | Description                                               |
|----------------------------------|-------------------|-----------------------------------------------------------|
| ConstMatrix                      | Dims              | Return the number of rows and columns of the matrix       |
| ConstMatrix                      | Equals            | Returns true if the two matrixs are equal                 |
| ConstMatrix                      | Table             | Converts matrix to a string                               |
| ConstMatrix                      | Int8At            | Returns the scalar at the given position as int8          |
| ConstMatrix                      | Int16At           | Returns the scalar at the given position as int16         |
| ConstMatrix                      | Int32At           | Returns the scalar at the given position as int32         |
| ConstMatrix                      | Int64At           | Returns the scalar at the given position as int64         |
| ConstMatrix                      | IntAt             | Returns the scalar at the given position as int           |
| ConstMatrix                      | Float32At         | Returns the scalar at the given position as Float32       |
| ConstMatrix                      | Float64At         | Returns the scalar at the given position as Float64       |
| ConstMatrix                      | ConstAt           | Returns the scalar at the given position as *ConstScalar* |
| ConstMatrix                      | ConstSlice        | Returns a slice as a constant matrix (*ConstMatrix*)      |
| ConstMatrix                      | ConstRow          | Returns the ith row as a *ConstVector*                    |
| ConstMatrix                      | ConstCol          | Returns the jth column as a *ConstVector*                 |
| ConstMatrix                      | ConstIterator     | Returns a constant iterator                               |
| ConstMatrix                      | AsConstVector     | Convert matrix to a vector of type *ConstVector*          |
| ConstMatrix                      | CloneConstMatrix  | Return a deep copy of the matrix as *ConstMatrix*         |
| ConstMatrix, Matrix              | At                | Return the scalar the given index                         |
| ConstMatrix, Matrix              | Reset             | Set all scalars to zero                                   |
| ConstMatrix, Matrix              | Set               | Set the value and derivatives of a scalar                 |
| ConstMatrix, Matrix              | Slice             | Return a slice of the matrix                              |
| ConstMatrix, Matrix              | Export            | Export matrix to file                                     |
| ConstMatrix, Matrix              | Permute           | Permute elements of the matrix                            |
| ConstMatrix, Matrix              | ReverseOrder      | Reverse the order of matrix elements                      |
| ConstMatrix, Matrix              | Sort              | Sort matrix elements                                      |
| ConstMatrix, Matrix              | AppendScalar      | Append a single scalar to the matrix                      |
| ConstMatrix, Matrix              | AppendMatrix      | Append another matrix                                     |
| ConstMatrix, Matrix              | Swap              | Swap two elements of the matrix                           |
| ConstMatrix, Matrix              | SwapRows          | Swap two rows                                             |
| ConstMatrix, Matrix              | SwapCols          | Swap two columns                                          |
| ConstMatrix, Matrix              | PermuteRows       | Permute rows                                              |
| ConstMatrix, Matrix              | PermuteCols       | Permute columns                                           |
| ConstMatrix, Matrix              | Row               | Returns a copy of the ith row as a *Vector*               |
| ConstMatrix, Matrix              | Col               | Returns a copy of the jth column a *Vector*               |
| ConstMatrix, Matrix              | T                 | Returns a transposed matrix                               |
| ConstMatrix, Matrix              | Tip               | Transpose in-place                                        |
| ConstMatrix, Matrix              | AsVector          | Convert matrix to a vector of type *Vector*               |
| ConstMatrix, Matrix              | Iterator          | Returns an iterator                                       |
| ConstMatrix, Matrix              | CloneMatrix       | Return a deep copy of the matrix as *Matrix*              |
| ConstMatrix, Matrix, MagicMatrix | MagicAt           | Returns the scalar at the given position as *MagicScalar* |
| ConstMatrix, Matrix, MagicMatrix | MagicSlice        | Resutns a slice as a magic matrix (*MagicMatrix*)         |
| ConstMatrix, Matrix, MagicMatrix | MagicT            | Returns a transposed matrix of type *MagicMatrix*         |
| ConstMatrix, Matrix, MagicMatrix | AppendMagicScalar | Append a single magic scalar                              |
| ConstMatrix, Matrix, MagicMatrix | AppendMagicMatrix | Append a magic matrix                                     |
| ConstMatrix, Matrix, MagicMatrix | CloneMagicMatrix  | Return a deep copy of the matrix as *MagicMatrix*         |

Matrices support the following linear algebra operations:

| Function | Description                      |
| -------- | -------------------------------- |
| MaddM    | Element-wise addition            |
| MsubM    | Element-wise substraction        |
| MmulM    | Element-wise multiplication      |
| MdivM    | Element-wise division            |
| MaddS    | Addition of a scalar             |
| MsubS    | Substraction of a scalar         |
| MmulS    | Multiplication with a scalar     |
| MdivS    | Division by a scalar             |
| MdotM    | Matrix product                   |
| Outer    | Outer product                    |

Methods, such as *VaddV* and *MaddM*, are generic and accept vector or matrix types that implement the respective *ConstVector* or *ConstMatrix* interface. However, opertions on interface types are much slower than on concrete types, which is why most vector and matrix types in *autodiff* also implement methods that operate on concrete types. For instance, *DenseFloat64Vector* implements a method called *VADDV* that takes as arguments two objects of type *DenseFloat64Vector*. Methods that operate on concrete types are always named in capital letters.

## Algorithms

The algorithms package contains more complex linear algebra and optimization routines:

| Package             | Description                                             |
| ------------------- | ------------------------------------------------------- |
| Adam                | Adam stochastic gradient method                         |
| bfgs                | Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm       |
| blahut              | Blahut algorithm (channel capacity)                     |
| cholesky            | Cholesky and LDL factorization                          |
| determinant         | Matrix determinants                                     |
| eigensystem         | Compute Eigenvalues and Eigenvectors                    |
| gaussJordan         | Gauss-Jordan algorithm                                  |
| gradientDescent     | Vanilla gradient desent algorithm                       |
| gramSchmidt         | Gram-Schmidt algorithm                                  |
| hessenbergReduction | Matrix Hessenberg reduction                             |
| lineSearch          | Line-search (satisfying the Wolfe conditions)           |
| matrixInverse       | Matrix inverse                                          |
| msqrt               | Matrix square root                                      |
| msqrtInv            | Inverse matrix square root                              |
| newton              | Newton's method (root finding and optimization)         |
| qrAlgorithm         | QR-Algorithm for computing Schur decompositions         |
| rprop               | Resilient backpropagation                               |
| svd                 | Singular Value Decomposition (SVD)                      |
| saga                | SAGA stochastic average gradient descent method         |

## Basic usage

Import the autodiff library with
```go
  import . "github.com/pbenner/autodiff"
```
A scalar holding the value *1.0* can be defined in several ways, i.e.
```go
  a := NullScalar(Real64Type)
  a.SetFloat64(1.0)
  b := NewReal64(1.0)
  c := NewFloat64(1.0)
```
*a* and *b* are both *MagicScalar*s, however *a* has type *Scalar* whereas *b* has type **Real64* which implements the *Scalar* interface. Variable *c* is of type *Float64* which cannot carry any derivatives. Basic operations such as additions are defined on all Scalars, i.e.
```go
  a.Add(a, b)
```
which stores the result of adding *a* and *b* in *a*. The *ConstFloat64* type allows to define float64 constants without allocation of additional memory. For instance
```go
  a.Add(a, ConstFloat64(1.0))
```
adds a constant value to *a* where a type cast is used to define the constant *1.0*.

To differentiate a function *f(x,y) = x y^3 + 4*, we define
```go
  f := func(x, y ConstScalar) MagicScalar {
    // compute f(x,y) = x*y^3 + 4
    z := NewReal64()
    z.Pow(y, ConstFloat64(3.0))
    z.Mul(z, x)
    z.Add(z, ConstFloat64(4.0))
    return z
  }
```
that accepts as arguments two *ConstScalar* variables and returns a *MagicScalar*. We first define two *MagicReal* variables
```go
  x := NewReal64(2)
  y := NewReal64(4)
```
that store the value at which the derivatives should be evaluated. Afterwards, *x* and *y* must be activated with
```go
  Variables(2, x, y)
```
where the first argument sets the order of the derivatives. *Variables()* should only be called once, as it allocates memory for the given magic variables. In this case, derivatives up to second order are computed. After evaluating *f*, i.e.
```go
  z := f(x, y)
```
the function value at *(x,y) = (2, 4)* can be retrieved with *z.GetFloat64()*. The first and second partial derivatives can be accessed with *z.GetDerivative(i)* and *z.GetHessian(i, j)*, where the arguments specify the index of the variable. For instance, the derivative of *f* with respect to *x* is returned by *z.GetDerivative(0)*, whereas the derivative with respect to *y* by *z.GetDerivative(1)*.

## Basic linear algebra

Vectors and matrices can be created with
```go
  v := NewDenseFloat64Vector([]float64{1,2})
  m := NewDenseFloat64Matrix([]float64{1,2,3,4}, 2, 2)

  v_ := NewDenseReal64Vector([]float64{1,2})
  m_ := NewDenseReal64Matrix([]float64{1,2,3,4}, 2, 2)
```
where *v* has length 2 and *m* is a 2x2 matrix. With
```go
  v := NullDenseFloat64Vector(2)
  m := NullDenseFloat64Matrix(2, 2)
```
all values are initially set to zero. Vector and matrix elements can be accessed with the *At*, *MagicAt* or *ConstAt* methods, which return a reference to the scalar implementing either a *Scalar*, *MagicScalar* or *ConstScalar*, i.e.
```go
  m.At(1,1).Add(v.ConstAt(0), v.ConstAt(1))
```
adds the first two values in *v* and stores the result in the lower right element of the matrix *m*. Autodiff supports basic linear algebra operations, for instance, the vector matrix product can be computed with
```go
  w := NullDenseFloat64Vector(2)
  w.MdotV(m, v)
```
where the result is stored in w. Other operations, such as computing the eigenvalues and eigenvectors of a matrix, require importing the respective package from the algorithm library, i.e.
```go
  import "github.com/pbenner/autodiff/algorithm/eigensystem"

  lambda, _, _ := eigensystem.Run(m)
```

## Examples

### Gradient descent

Compare vanilla gradient descent with resilient backpropagation
```go
  import . "github.com/pbenner/autodiff"
  import   "github.com/pbenner/autodiff/algorithm/gradientDescent"
  import   "github.com/pbenner/autodiff/algorithm/rprop"

  f := func(x_ ConstVector) MagicScalar {
    x := x_.ConstAt(0)
    // x^4 - 3x^3 + 2
    r := NewReal64()
    s := NewReal64()
    r.Pow(x.ConstAt(0), ConstFloat64(4.0)
    s.Mul(ConstFloat64(3.0), s.Pow(x, ConstFloat64(3.0)))
    r.Add(ConstFloat64(2.0), r.Add(r, s))
    return r
  }
  x0 := NewDenseFloat64Vector([]float64{8})
  // vanilla gradient descent
  xn1, _ := gradientDescent.Run(f, x0, 0.0001, gradientDescent.Epsilon{1e-8})
  // resilient backpropagation
  xn2, _ := rprop.Run(f, x0, 0.0001, 0.4, rprop.Epsilon{1e-8})
```
![Gradient descent](demo/example1/example1.png)


### Matrix inversion

Compute the inverse *r* of a matrix *m* by minimizing the Frobenius norm *||mb - I||*
```go
  import . "github.com/pbenner/autodiff"
  import   "github.com/pbenner/autodiff/algorithm/rprop"

  // define matrix r
  m := NewDenseFloat64Matrix([]float64{1,2,3,4}, 2, 2)
  // create identity matrix I
  I := NullDenseFloat64Matrix(2, 2)
  I.SetIdentity()

  // magic variables for computing the Frobenius norm and its derivative
  t := NewDenseReal64Matrix(2, 2)
  s := NewReal64()
  // objective function
  f := func(x ConstVector) MagicScalar {
    t.Set(x)
    s.Mnorm(t.MsubM(t.MmulM(m, t), I))
    return s
  }
  r, _ := rprop.Run(f, r.GetValues(), 0.01, 0.1, rprop.Epsilon{1e-12})
```

### Newton's method

Find the root of a function *f* with initial value *x0 = (1,1)*

```go
  import . "github.com/pbenner/autodiff"
  import   "github.com/pbenner/autodiff/algorithm/newton"

  t := NullReal64()

  f := func(x ConstVector) MagicVector {
    x1 := x.ConstAt(0)
    x2 := x.ConstAt(1)
    y  := NullDenseReal64Vector(2)
    y1 := y.At(0)
    y2 := y.At(1)
    // y1 = x1^2 + x2^2 - 6
    t .Pow(x1, ConstFloat64(2.0))
    y1.Add(y1, t)
    t .Pow(x2, ConstFloat64(2.0))
    y1.Add(y1, t)
    y1.Sub(y1, ConstFloat64(6.0))
    // y2 = x1^3 - x2^2
    t .Pow(x1, ConstFloat64(3.0))
    y2.Add(y2, t)
    t .Pow(x2, ConstFloat64(2.0))
    y2.Sub(y2, t)

    return y
  }

  x0    := NewDenseFloat64Vector([]float64{1,1})
  xn, _ := newton.RunRoot(f, x0, newton.Epsilon{1e-8})
```

### Minimize Rosenbrock's function
Compare Adam, Newton's method, BFGS and Rprop for minimizing Rosenbrock's function

```go
  import   "fmt"

  import . "github.com/pbenner/autodiff"
  import   "github.com/pbenner/autodiff/algorithm/adam"
  import   "github.com/pbenner/autodiff/algorithm/rprop"
  import   "github.com/pbenner/autodiff/algorithm/bfgs"
  import   "github.com/pbenner/autodiff/algorithm/newton"

  f := func(x ConstVector) (MagicScalar, error) {
    // f(x1, x2) = (a - x1)^2 + b(x2 - x1^2)^2
    // a = 1
    // b = 100
    // minimum: (x1,x2) = (a, a^2)
    a := ConstFloat64(  1.0)
    b := ConstFloat64(100.0)
    c := ConstFloat64(  2.0)
    s := NullReal64()
    t := NullReal64()
    s.Pow(s.Sub(a, x.ConstAt(0)), c)
    t.Mul(b, t.Pow(t.Sub(x.ConstAt(1), t.Mul(x.ConstAt(0), x.ConstAt(0))), c))
    s.Add(s, t)
    return s, nil
  }
  hook_adam := func(x, gradient ConstVector, hessian ConstMatrix, y ConstScalar) bool {
    fmt.Println("x       :", x)
    fmt.Println("gradient:", gradient)
    fmt.Println("y       :", y)
    fmt.Println()
    return false
  }
  hook_rprop := func(gradient, step []float64, x ConstVector, y ConstScalar) bool {
    fmt.Println("x       :", x)
    fmt.Println("gradient:", gradient)
    fmt.Println("y       :", y)
    fmt.Println()
    return false
  }
  hook_bfgs := func(x, gradient ConstVector, y ConstScalar) bool {
    fmt.Println("x       :", x)
    fmt.Println("gradient:", gradient)
    fmt.Println("y       :", y)
    fmt.Println()
    return false
  }
  hook_newton := func(x, gradient ConstVector, hessian ConstMatrix, y ConstScalar) bool {
    fmt.Println("x       :", x)
    fmt.Println("gradient:", gradient)
    fmt.Println("y       :", y)
    fmt.Println()
    return false
  }

  x0 := NewDenseFloat64Vector([]float64{-0.5, 2})

  adam.Run(f, x0,
    adam.StepSize{0.1},
    adam.Hook{hook_adam},
    adam.Epsilon{1e-10})

  rprop.Run(f, x0, 0.05, []float64{1.2, 0.8},
    rprop.Hook{hook_rprop},
    rprop.Epsilon{1e-10})

  bfgs.Run(f, x0,
    bfgs.Hook{hook_bfgs},
    bfgs.Epsilon{1e-10})

  newton.RunMin(f, x0,
    newton.HookMin{hook_newton},
    newton.Epsilon{1e-8},
    newton.HessianModification{"LDL"})
```
![Gradient descent](demo/rosenbrock/rosenbrock.png)

### Constrained optimization

Maximize the function *f(x, y) = x + y* subject to *x^2 + y^2 = 1* by finding the critical point of the corresponding Lagrangian

```go
  import . "github.com/pbenner/autodiff"
  import   "github.com/pbenner/autodiff/algorithm/newton"

  z := NullReal64()
  t := NullReal64()
  // define the Lagrangian
  f := func(x_ ConstVector) (MagicScalar, error) {
    // z = x + y + lambda(x^2 + y^2 - 1)
    x      := x_.ConstAt(0)
    y      := x_.ConstAt(1)
    lambda := x_.ConstAt(2)
    z.Reset()
    t.Pow(x, ConstFloat64(2.0))
    z.Add(z, t)
    t.Pow(y, ConstFloat64(2.0))
    z.Add(z, t)
    z.Sub(z, ConstFloat64(1.0))
    z.Mul(z, lambda)
    z.Add(z, y)
    z.Add(z, x)

    return z, nil
  }
  // initial value
  x0    := NewDenseFloat64Vector([]float64{3,  5, 1})
  // run Newton's method
  xn, _ := newton.RunCrit(
      f, x0,
      newton.Epsilon{1e-8})
```
