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
  Balance         bool
  Epsilon         float64
  L1Reg           float64
  L2Reg           float64
  TiReg           float64
  MaxIterations   int
  ClassWeights [2]float64
  Seed            int64
  Hook            func(x ConstVector, step ConstScalar, i int) bool
}

/* -------------------------------------------------------------------------- */

func NewLogisticRegression(n int, sparse bool) (*LogisticRegression, error) {
  r := LogisticRegression{}
  r.logisticRegression.Theta = NullDenseBareRealVector(n)
  r.Epsilon         = 1e-5
  r.MaxIterations   = int(^uint(0) >> 1)
  r.ClassWeights[0] = 1.0
  r.ClassWeights[1] = 1.0
  r.sparse          = sparse
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

// x_i = (1.0, x_i1, x_i2, ..., x_im, class_label)
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
      if x[i].ValueAt(0) != 1.0 {
        return fmt.Errorf("first element of data vector must be set to one")
      }
      t := x[i].ConstSlice(0, x[i].Dim()-1)
      switch a := t.(type) {
      case SparseConstRealVector:
        obj.x_sparse = append(obj.x_sparse, a)
      default:
        obj.x_sparse = append(obj.x_sparse, AsSparseConstRealVector(t))
      }
    }
  } else {
    for i, _ := range x {
      if x[i].Dim() != x[0].Dim() {
        return fmt.Errorf("data has inconsistent dimensions")
      }
      if x[i].ValueAt(0) != 1.0 {
        return fmt.Errorf("first element of data vector must be set to one")
      }
      t := x[i].ConstSlice(0, x[i].Dim()-1)
      switch a := t.(type) {
      case DenseConstRealVector:
        obj.x_dense = append(obj.x_dense, a)
      default:
        obj.x_dense = append(obj.x_dense, AsDenseConstRealVector(t))
      }
    }
  }
  for i, _ := range x {
    v := x[i].ValueAt(x[i].Dim()-1)
    switch v {
    case 1.0: obj.c = append(obj.c, true )
    case 0.0: obj.c = append(obj.c, false)
    default : return fmt.Errorf("invalid class label `%f'", v)
    }
    obj.x = append(obj.x, x[i])
  }
  if obj.Balance {
    n1 := 0
    n0 := 0
    for i := 0; i < len(obj.c); i++ {
      switch obj.c[i] {
      case true : n1++
      case false: n0++
      }
    }
    obj.ClassWeights[1] = 1.0/float64(n1)
    obj.ClassWeights[0] = 1.0/float64(n0)
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
  var jitUpdate saga.JitUpdateType
  switch {
  case obj.L1Reg != 0.0 || (obj.L2Reg == 0.0 && obj.TiReg == 0.0):
    if obj.sparse {
      // use specialized saga implementation
      if r, err := sagaLogisticRegressionL1(saga.Objective1Sparse(obj.f_sparse), len(obj.x_sparse), obj.Theta,
        saga.L1Regularization{obj.L1Reg},
        saga.Gamma           {obj.stepSize},
        saga.Epsilon         {obj.Epsilon},
        saga.MaxIterations   {obj.MaxIterations},
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
  case obj.L2Reg != 0.0: proxop = proximalWrapper{&saga.ProximalOperatorL2{obj.L2Reg}}
  case obj.TiReg != 0.0: proxop = proximalWrapper{&saga.ProximalOperatorTi{obj.TiReg}}
  }
  if obj.sparse {
    if r, err := saga.Run(saga.Objective1Sparse(obj.f_sparse), len(obj.x_sparse), obj.Theta,
      saga.Hook            {obj.Hook},
      saga.Gamma           {obj.stepSize},
      saga.Epsilon         {obj.Epsilon},
      saga.MaxIterations   {obj.MaxIterations},
      saga.Seed            {obj.Seed},
      saga.ProximalOperator{proxop},
      saga.JitUpdate       {jitUpdate}); err != nil {
      return err
    } else {
      obj.SetParameters(r)
    }
  } else {
    if r, err := saga.Run(saga.Objective1Dense(obj.f_dense), len(obj.x_dense), obj.Theta,
      saga.Hook            {obj.Hook},
      saga.Gamma           {obj.stepSize},
      saga.Epsilon         {obj.Epsilon},
      saga.MaxIterations   {obj.MaxIterations},
      saga.Seed            {obj.Seed},
      saga.ProximalOperator{proxop},
      saga.JitUpdate       {jitUpdate}); err != nil {
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
    // skip first element
    if it.Ok() {
      it.Next()
    }
    for ; it.Ok(); it.Next() {
      // skip last element
      if it.Index() == obj.x[i].Dim()-1 {
        break
      }
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

type jitUpdateWrapper struct {
  saga.JitUpdateType
}

func (obj jitUpdateWrapper) Update(x, y BareReal, k, m int) BareReal {
  // do not regularize intercept
  if k == 0 {
    return x - BareReal(m)*y
  } else {
    return obj.JitUpdateType.Update(x, y, k, m)
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
    w = ConstReal(obj.ClassWeights[1]*(math.Exp(r) - 1.0))
  } else {
    w = ConstReal(obj.ClassWeights[0]*(math.Exp(r)))
  }
  return y, w, x[i], nil
}

func (obj *LogisticRegression) f_sparse(i int, theta DenseBareRealVector) (ConstReal, ConstReal, SparseConstRealVector, error) {
  x := obj.x_sparse
  y := ConstReal(0.0)
  w := ConstReal(0.0)
  if len(theta) == 0 {
    return y, w, x[i], nil
  }
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
    w = ConstReal(obj.ClassWeights[1]*(math.Exp(r) - 1.0))
  } else {
    w = ConstReal(obj.ClassWeights[0]*(math.Exp(r)))
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

type sagaJitUpdateL1 struct {
  saga.JitUpdateL1
}

func (obj sagaJitUpdateL1) Update(x, y BareReal, k, m int) BareReal {
  // do not regularize intercept
  if k == 0 {
    return x - BareReal(m)*y
  } else {
    return obj.JitUpdateL1.Update(x, y, k, m)
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

  x0 := AsDenseBareRealVector(x)
  x1 := AsDenseBareRealVector(x)
  xk := make([]int,  x.Dim())
  xs := make([]bool, n)
  ns := 0
  cumulative_sums := NullDenseBareRealVector(n)

  // length of gradient
  d := x.Dim()
  // gradient
  var g1 saga.GradientJit
  var g2 saga.GradientJit

  // some constants
  t_n := BareReal(0.0)
  t_g := BareReal(gamma.Value)
  t_l := BareReal(l1reg.Value)*t_g/BareReal(n)

  // jit update function
  jit := sagaJitUpdateL1{}
  jit.SetLambda(float64(t_l))

  // sum of gradients
  s := NullDenseBareRealVector(d)
  // initialize s and d
  dict := make([]saga.GradientJit, n)
  for i := 0; i < n; i++ {
    if _, w, g, err := f(i, nil); err != nil {
      return nil, err
    } else {
      dict[i].Set(w, g)
    }
  }
  g := rand.New(rand.NewSource(seed.Value))

  for epoch := 0; epoch < maxIterations.Value; epoch++ {
    for i_ := 0; i_ < n; i_++ {
      j := i_
      if seed.Value != -1 {
        j = g.Intn(n)
      }
      if !xs[j] {
        ns += 1
        t_n = BareReal(ns)
      }
      if i_ == 0 {
        cumulative_sums[0 ] = t_g/t_n
      } else {
        cumulative_sums[i_] = cumulative_sums[i_-1] + t_g/t_n
      }
      // get old gradient
      g1 = dict[j]
      // perform jit updates for all x_i where g_i != 0
      for _, k := range g1.G.GetSparseIndices() {
        if m := i_-xk[k]; m > 0 {
          cum_sum := cumulative_sums[i_-1]
          if xk[k] != 0 {
            cum_sum -= cumulative_sums[xk[k]-1]
          }
          x1[k] = jit.Update(x1[k], cum_sum*s[k]/BareReal(m), k, m)
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
        x1[k] = x1[k] - t_g*(1.0 - 1.0/t_n)*c*BareReal(v[i])
        xk[k] = i_
      }
      if !xs[j] {
        xs[j] = true
        g2.Add(s)
      } else {
        // update gradient avarage
        g1.Update(g2, s)
      }
      // update dictionary
      dict[j].Set(g2.W, g2.G)
    }
    // compute missing updates of x1
    for k := 0; k < x1.Dim(); k++ {
      if m := n-xk[k]; m > 0 {
        cum_sum := cumulative_sums[n-1]
        if xk[k] != 0 {
          cum_sum -= cumulative_sums[xk[k]-1]
        }
        x1[k] = jit.Update(x1[k], cum_sum*s[k]/BareReal(m), k, m)
      }
      // reset xk
      xk[k] = 0
    }
    if stop, delta, err := saga.EvalStopping(x0, x1, epsilon.Value*gamma.Value); stop {
      return x1, err
    } else {
      // execute hook if available
      if hook.Value != nil && hook.Value(x1, ConstReal(delta), epoch) {
        break
      }
    }
    x0.SET(x1)
  }
  return x1, nil
}
