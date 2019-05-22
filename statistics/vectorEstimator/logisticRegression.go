/* Copyright (C) 2019 Philipp Benner
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package vectorEstimator

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"
import   "math/rand"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/saga"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import . "github.com/pbenner/autodiff/logarithmetic"

import . "github.com/pbenner/threadpool"


/* -------------------------------------------------------------------------- */

type logisticRegression struct {
  Theta DenseBareRealVector
}

/* -------------------------------------------------------------------------- */

func (obj logisticRegression) Dim() int {
  return len(obj.Theta)-1
}

func (obj logisticRegression) LogPdfDense(x DenseConstRealVector) float64 {
  // set r to first element of theta
  r := float64(obj.Theta[0])
  n := x.Dim()
  for i := 1; i < n; i++ {
    r += float64(x[i])*float64(obj.Theta[i])
  }
  return -LogAdd(0.0, -r)
}

func (obj logisticRegression) LogPdfSparse(v SparseConstRealVector) float64 {
  x     := v.GetSparseValues ()
  index := v.GetSparseIndices()
  // set r to first element of theta
  r := float64(obj.Theta[0])
  // loop over x
  i := 0
  n := len(index)
  // skip first element
  if index[i] == 0 {
    i++
  }
  for ; i < n; i++ {
    r += float64(x[i])*float64(obj.Theta[index[i]])
  }
  return -LogAdd(0.0, -r)
}

/* -------------------------------------------------------------------------- */

type LogisticRegression struct {
  logisticRegression
  sparse     bool
  n          int
  x_sparse []SparseConstRealVector
  x_dense  [] DenseConstRealVector
  x        []ConstVector
  c        []bool
  stepSize   float64
  // optional parameters
  Epsilon    float64
  L1Reg      float64
  L2Reg      float64
  TiReg      float64
  Seed       int64
  Hook       func(x ConstVector, step ConstScalar, i int) bool
}

/* -------------------------------------------------------------------------- */

func NewLogisticRegression(index []int, theta []float64, n int) (*LogisticRegression, error) {
  r := LogisticRegression{}
  r.Epsilon = 1e-8
  if index == nil {
    if len(theta) != n {
      return nil, fmt.Errorf("theta has invalid dimension")
    }
    r.sparse = false
    r.logisticRegression.Theta = NewDenseBareRealVector(theta)
  } else {
    r.sparse = true
    r.logisticRegression.Theta = AsDenseBareRealVector(NewSparseBareRealVector(index, theta, n))
  }
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) Clone() *LogisticRegression {
  r := LogisticRegression{}
  // copy data and optional arguments
  r  = *obj
  r.logisticRegression.Theta = obj.logisticRegression.Theta.Clone()
  return &r
}

func (obj *LogisticRegression) CloneVectorEstimator() VectorEstimator {
  return obj.Clone()
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) ScalarType() ScalarType {
  return BareRealType
}

func (obj *LogisticRegression) GetParameters() Vector {
  return obj.Theta
}

func (obj *LogisticRegression) SetParameters(x Vector) error {
  obj.Theta = AsDenseBareRealVector(x)
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) GetData() ([]ConstVector, int) {
  return obj.x, obj.n
}

// x_i = (class_label, 1.0, x_i1, x_i2, ..., x_im)
func (obj *LogisticRegression) SetData(x []ConstVector, n int) error {
  obj.n = n
  // reset data
  obj.x_sparse = nil
  obj.x_dense  = nil
  obj.x        = nil
  obj.c        = nil
  if len(x) == 0 {
    return nil
  }
  if k := obj.logisticRegression.Dim()+2; x[0].Dim() != k {
    return fmt.Errorf("LogisticRegression: data has invalid dimension: got data of dimension `%d' but expected dimension `%d'", x[0].Dim(), k)
  }
  if obj.sparse {
    for i, _ := range x {
      if x[i].Dim() != x[0].Dim() {
        return fmt.Errorf("data has inconsistent dimensions")
      }
      if x[i].ValueAt(1) != 1.0 {
        return fmt.Errorf("second element of data vector must be set to one")
      }
      t := x[i].ConstSlice(1, x[1].Dim())
      switch a := t.(type) {
      case SparseConstRealVector:
        obj.x_sparse = append(obj.x_sparse, a)
      default:
        obj.x_sparse = append(obj.x_sparse, AsSparseConstRealVector(t))
      }
      obj.x = append(obj.x, obj.x_sparse[i])
    }
  } else {
    for i, _ := range x {
      if x[i].Dim() != x[0].Dim() {
        return fmt.Errorf("data has inconsistent dimensions")
      }
      if x[i].ValueAt(1) != 1.0 {
        return fmt.Errorf("second element of data vector must be set to one")
      }
      t := x[i].ConstSlice(1, x[1].Dim())
      switch a := t.(type) {
      case DenseConstRealVector:
        obj.x_dense = append(obj.x_dense, a)
      default:
        obj.x_dense = append(obj.x_dense, AsDenseConstRealVector(t))
      }
      obj.x = append(obj.x, obj.x_dense[i])
    }
  }
  for i, _ := range x {
    switch x[i].ConstAt(0).GetValue() {
    case 1.0: obj.c = append(obj.c, true )
    case 0.0: obj.c = append(obj.c, false)
    default : return fmt.Errorf("invalid class label `%v'", x[i].ConstAt(0))
    }
  }
  obj.setStepSize()
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) Estimate(gamma ConstVector, p ThreadPool) error {
  if gamma != nil {
    panic("internal error")
  }
  { m := 0
    if obj.L1Reg != 0.0 { m++ }
    if obj.L2Reg != 0.0 { m++ }
    if obj.TiReg != 0.0 { m++ }
    if m > 1 {
      return fmt.Errorf("multiple regularizations are not supported")
    }
  }
  var proxop    saga.ProximalOperatorType
  var proxopjit saga.ProximalOperatorJitType
  switch {
  case obj.L1Reg != 0.0:
    if obj.sparse {
      // use specialized saga implementation
      if r, err := sagaLogisticRegressionL1(saga.Objective1Sparse(obj.f_sparse), len(obj.x_sparse), obj.Theta,
        saga.L1Regularization{obj.L1Reg},
        saga.Gamma           {obj.stepSize},
        saga.Epsilon         {obj.Epsilon},
        saga.MaxIterations   {int(^uint(0) >> 1)},
        saga.Hook            {obj.Hook},
        saga.Seed            {obj.Seed}); err != nil {
        return err
      } else {
        obj.SetParameters(r)
        return nil
      }
    } else {
      proxop = proximalWrapper{&saga.ProximalOperatorL1{obj.L1Reg}}
    }
  case obj.L2Reg != 0.0: proxop = proximalWrapper   {&saga.ProximalOperatorL2   {obj.L2Reg}}
  case obj.TiReg != 0.0: proxop = proximalWrapper   {&saga.ProximalOperatorTi   {obj.TiReg}}
  }
  if obj.sparse {
    if r, err := saga.Run(saga.Objective1Sparse(obj.f_sparse), len(obj.x_sparse), obj.Theta,
      saga.Hook   {obj.Hook},
      saga.Gamma  {obj.stepSize},
      saga.Epsilon{obj.Epsilon},
      saga.Seed   {obj.Seed}); err != nil {
      return err
    } else {
      obj.SetParameters(r)
    }
  } else {
    if r, err := saga.Run(saga.Objective1Dense(obj.f_dense), len(obj.x_dense), obj.Theta,
      saga.Hook   {obj.Hook},
      saga.Gamma  {obj.stepSize},
      saga.Epsilon{obj.Epsilon},
      saga.Seed   {obj.Seed},
      saga.ProximalOperator   {proxop},
      saga.ProximalOperatorJit{proxopjit}); err != nil {
      return err
    } else {
      obj.SetParameters(r)
    }
  }
  return nil
}

func (obj *LogisticRegression) EstimateOnData(x []ConstVector, gamma ConstVector, p ThreadPool) error {
  if err := obj.SetData(x, len(x)); err != nil {
    return err
  }
  return obj.Estimate(gamma, p)
}

func (obj *LogisticRegression) GetEstimate() (VectorPdf, error) {
  return vectorDistribution.NewLogisticRegression(obj.Theta)
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) setStepSize() {
  max_squared_sum := 0.0
  for i, _ := range obj.x {
    r  := 0.0
    it := obj.x[i].ConstIterator()
    // skep first element (class)
    if it.Ok() {
      it.Next()
    }
    for ; it.Ok(); it.Next() {
      r += it.GetConst().GetValue()*it.GetConst().GetValue()
    }
    if r > max_squared_sum {
      max_squared_sum = r
    }
  }
  L  := (0.25*(max_squared_sum + 1.0) + obj.L2Reg/float64(obj.n))
  obj.stepSize = 1.0/(2.0*L + math.Min(2.0*obj.L2Reg, L))
}

/* -------------------------------------------------------------------------- */

type proximalWrapper struct {
  saga.ProximalOperatorType
}

func (obj proximalWrapper) Eval(x DenseBareRealVector, w DenseBareRealVector, t *BareReal) {
  obj.ProximalOperatorType.Eval(x, w, t)
  // do not regularize intercept
  x.AT(0).SET(w.AT(0))
}

/* -------------------------------------------------------------------------- */

type proximalWrapperJit struct {
  saga.ProximalOperatorJitType
}

func (obj proximalWrapperJit) Eval(x *BareReal, w *BareReal, i, n int, t *BareReal) {
  // do not regularize intercept
  if i != 0 {
    obj.ProximalOperatorJitType.Eval(x, w, i, n, t)
  } else {
    x.SET(w)
  }
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) f_dense(i int, theta DenseBareRealVector) (ConstReal, ConstReal, DenseConstRealVector, error) {
  x := obj.x_dense
  y := ConstReal(0.0)
  w := ConstReal(0.0)
  if i >= len(x) {
    return y, w, x[i], fmt.Errorf("index out of bounds")
  }
  obj.logisticRegression.Theta = theta

  r := obj.logisticRegression.LogPdfDense(x[i])

  if math.IsNaN(r) {
    return y, w, x[i], fmt.Errorf("NaN value detected")
  }
  y = ConstReal(r)
  if obj.c[i] {
    w = ConstReal(math.Exp(r) - 1.0)
  } else {
    w = ConstReal(math.Exp(r))
  }
  return y, w, x[i], nil
}

func (obj *LogisticRegression) f_sparse(i int, theta DenseBareRealVector) (ConstReal, ConstReal, SparseConstRealVector, error) {
  x := obj.x_sparse
  y := ConstReal(0.0)
  w := ConstReal(0.0)
  if i >= len(x) {
    return y, w, x[i], fmt.Errorf("index out of bounds")
  }
  obj.logisticRegression.Theta = theta

  r := obj.logisticRegression.LogPdfSparse(x[i])

  if math.IsNaN(r) {
    return y, w, x[i], fmt.Errorf("NaN value detected")
  }
  y = ConstReal(r)
  if obj.c[i] {
    w = ConstReal(math.Exp(r) - 1.0)
  } else {
    w = ConstReal(math.Exp(r))
  }
  return y, w, x[i], nil
}

/* -------------------------------------------------------------------------- */

func sagaProxopL1(w, lambda BareReal, i int, n BareReal) BareReal {
  if i == 0 {
    return w
  }
  // sign(wi)*max{|wi| - n*lambda}
  if w < 0.0 {
    if l := n*lambda; -w < l {
      return BareReal(0.0)
    } else {
      return w + l
    }
  } else {
    if l := n*lambda;  w < l {
      return BareReal(0.0)
    } else {
      return w - l
    }
  }
}

func sagaLogisticRegressionL1(
  f saga.Objective1Sparse,
  n int,
  x DenseBareRealVector,
  l1reg saga.L1Regularization,
  gamma saga.Gamma,
  epsilon saga.Epsilon,
  maxIterations saga.MaxIterations,
  hook saga.Hook,
  seed saga.Seed) (DenseBareRealVector, error) {

  xs := AsDenseBareRealVector(x)
  x1 := AsDenseBareRealVector(x)
  xk := make([]int, x.Dim())

  // length of gradient
  d := x.Dim()
  // gradient
  var g1 saga.GradientJit
  var g2 saga.GradientJit

  // temporary variables
  t1 := BareReal(0.0)
  // some constants
  t_n := BareReal(n)
  t_g := BareReal(gamma.Value)
  t_l := BareReal(l1reg.Value)*t_g/t_n

  // sum of gradients
  s := NullDenseBareRealVector(d)
  // initialize s and d
  dict := make([]saga.GradientJit, n)
  for i := 0; i < n; i++ {
    if _, w, g, err := f(i, x1); err != nil {
      return nil, err
    } else {
      dict[i].Set(w, g)
      dict[i].Add(s)
    }
  }
  g := rand.New(rand.NewSource(seed.Value))

  for epoch := 0; epoch < maxIterations.Value; epoch++ {
    for i_ := 1; i_ < n+1; i_++ {
      j := g.Intn(n)

      // get old gradient
      g1 = dict[j]
      // perform jit updates for all x_i where g_i != 0
      for _, k := range g1.G.GetSparseIndices() {
        if m := i_ - xk[k]; m > 1 {
          t1 = x1[k] - BareReal(m-1)*t_g*s[k]/t_n
          x1[k] = sagaProxopL1(t1, t_l, k, BareReal(m-1))
        }
      }
      // evaluate objective function
      if _, w, g, err := f(j, x1); err != nil {
        return x1, err
      } else {
        g2.Set(w, g)
      }
      c := BareReal(g2.W - g1.W)
      v := g1.G.GetSparseValues()
      for i, k := range g1.G.GetSparseIndices() {
        t1 = x1[k] - t_g*(c*BareReal(v[i]) + s[k]/t_n)
        x1[k] = sagaProxopL1(t1, t_l, k, BareReal(1))
        xk[k] = i_
      }
      // update gradient avarage
      g1.Update(g2, s)

      // update dictionary
      dict[j].Set(g2.W, g2.G)
    }
    // compute missing updates of x1
    for k := 0; k < x1.Dim(); k++ {
      if m := n - xk[k]; m > 0 {
        t1 = x1[k] - BareReal(m)*t_g*s[k]/t_n
        x1[k] = sagaProxopL1(t1, t_l, k, BareReal(m))
      }
      // reset xk
      xk[k] = 0
    }
    if stop, delta, err := saga.EvalStopping(xs, x1, epsilon.Value*gamma.Value); stop {
      return x1, err
    } else {
      // execute hook if available
      if hook.Value != nil && hook.Value(x1, ConstReal(delta), epoch) {
        break
      }
    }
    xs.SET(x1)
  }
  return x1, nil
}
