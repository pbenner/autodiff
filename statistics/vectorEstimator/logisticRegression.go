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
  AutoReg         int
  L2Reg           float64
  TiReg           float64
  StepSizeFactor  float64
  MaxIterations   int
  ClassWeights [2]float64
  Seed            int64
  Hook            func(x ConstVector, step ConstScalar, i int) bool
  sagaLogisticRegressionL1state
}

/* -------------------------------------------------------------------------- */

func NewLogisticRegression(n int, sparse bool) (*LogisticRegression, error) {
  r := LogisticRegression{}
  r.logisticRegression.Theta = NullDenseBareRealVector(n)
  r.Epsilon         = 1e-5
  r.MaxIterations   = int(^uint(0) >> 1)
  r.ClassWeights[0] = 1.0
  r.ClassWeights[1] = 1.0
  r.StepSizeFactor  = 1.0
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
  if len(x) == 0 {
    return nil
  }
  if k := obj.logisticRegression.Dim()+2; x[0].Dim() != k {
    return fmt.Errorf("LogisticRegression: data has invalid dimension: got data of dimension `%d' but expected dimension `%d'", x[0].Dim(), k)
  }
  if obj.sparse {
    x_sparse := make([]ConstVector, len(x))
    for i, _ := range x {
      if x[i].Dim() != x[0].Dim() {
        return fmt.Errorf("data has inconsistent dimensions")
      }
      t := x[i].ConstSlice(0, x[i].Dim()-1)
      switch a := t.(type) {
      case SparseConstRealVector:
        x_sparse[i] = a
      default:
        x_sparse[i] = AsSparseConstRealVector(t)
      }
    }
    obj.SetSparseData(x_sparse, nil, n)
    obj.setStepSize()
  } else {
    x_dense := make([]ConstVector, len(x))
    for i, _ := range x {
      if x[i].Dim() != x[0].Dim() {
        return fmt.Errorf("data has inconsistent dimensions")
      }
      t := x[i].ConstSlice(0, x[i].Dim()-1)
      switch a := t.(type) {
      case DenseConstRealVector:
        x_dense[i] = a
      default:
        x_dense[i] = AsDenseConstRealVector(t)
      }
    }
    obj.SetDenseData(x_dense, nil, n)
    obj.setStepSize()
  }
  obj.c = make([]bool       , len(x))
  obj.x = make([]ConstVector, len(x))
  for i, _ := range x {
    switch a := x[i].(type) {
    case SparseConstRealVector:
      // do not use ValueAt to prevent that an index
      // for the sparse vector is constructed
      if j, v := a.First(); j != 0 || v != 1.0 {
        return fmt.Errorf("first element of data vector must be set to one")
      }
      if j, v := a.Last (); j != a.Dim()-1 {
        // last entry is not the class label =>
        // class is zero
        obj.c[i] = false
      } else {
        switch v {
        case 1.0: obj.c[i] = true
        case 0.0: obj.c[i] = false
        default : return fmt.Errorf("invalid class label `%f'", v)
        }
      }
    default:
      if x[i].ValueAt(0) != 1.0 {
        return fmt.Errorf("first element of data vector must be set to one")
      }
      v := x[i].ValueAt(x[i].Dim()-1)
      switch v {
      case 1.0: obj.c[i] = true
      case 0.0: obj.c[i] = false
      default : return fmt.Errorf("invalid class label `%f'", v)
      }
    }
    obj.x[i] = x[i]
  }
  obj.setLabels(obj.c)
  return nil
}

func (obj *LogisticRegression) setLabels(c []bool) {
  obj.c = c
  if obj.Balance {
    n1 := 0
    n0 := 0
    for i := 0; i < len(obj.c); i++ {
      switch obj.c[i] {
      case true : n1++
      case false: n0++
      }
    }
    obj.ClassWeights[1] = float64(n0+n1)/float64(2*n1)
    obj.ClassWeights[0] = float64(n0+n1)/float64(2*n0)
  }
}

func (obj *LogisticRegression) SetSparseData(x []ConstVector, c []bool, n int) error {
  obj.n        = n
  obj.x        = nil
  obj.x_sparse = make([]SparseConstRealVector, len(x))
  obj.x_dense  = nil
  obj.sparse   = true
  for i, _ := range x {
    switch a := x[i].(type) {
    case SparseConstRealVector:
      obj.x_sparse[i] = a
    default:
      return fmt.Errorf("data is not of type SparseConstRealVector")
    }
  }
  obj.setStepSize()
  obj.setLabels(c)
  return nil
}

func (obj *LogisticRegression) SetDenseData(x []ConstVector, c []bool, n int) error {
  obj.n        = n
  obj.x        = nil
  obj.x_sparse = nil
  obj.x_dense  = make([]DenseConstRealVector, len(x))
  obj.sparse   = false
  for i, _ := range x {
    switch a := x[i].(type) {
    case DenseConstRealVector:
      obj.x_dense[i] = a
    default:
      return fmt.Errorf("data is not of type DenseConstRealVector")
    }
  }
  obj.setStepSize()
  obj.setLabels(c)
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
  case obj.sparse && obj.L2Reg == 0.0 && obj.TiReg == 0.0:
    // use specialized saga implementation
    if r, s, err := obj.sagaLogisticRegressionL1(saga.Objective1Sparse(obj.f_sparse), len(obj.x_sparse), obj.Theta,
      saga.L1Regularization{obj.L1Reg},
      saga.AutoReg         {obj.AutoReg},
      saga.Gamma           {obj.stepSize},
      saga.Epsilon         {obj.Epsilon},
      saga.MaxIterations   {obj.MaxIterations},
      saga.Hook            {obj.Hook},
      saga.Seed            {obj.Seed}); err != nil {
      return err
    } else {
      obj.Seed = s
      obj.SetParameters(r)
      return nil
    }
  case obj.L1Reg != 0.0: proxop = proximalWrapper{&saga.ProximalOperatorL1{obj.L1Reg}}
  case obj.L2Reg != 0.0: proxop = proximalWrapper{&saga.ProximalOperatorL2{obj.L2Reg}}
  case obj.TiReg != 0.0: proxop = proximalWrapper{&saga.ProximalOperatorTi{obj.TiReg}}
  }
  if obj.sparse {
    if r, s, err := saga.Run(saga.Objective1Sparse(obj.f_sparse), len(obj.x_sparse), obj.Theta,
      saga.Hook            {obj.Hook},
      saga.Gamma           {obj.stepSize},
      saga.Epsilon         {obj.Epsilon},
      saga.MaxIterations   {obj.MaxIterations},
      saga.Seed            {obj.Seed},
      saga.ProximalOperator{proxop},
      saga.JitUpdate       {jitUpdate}); err != nil {
      return err
    } else {
      obj.Seed = s
      obj.SetParameters(r)
    }
  } else {
    if r, s, err := saga.Run(saga.Objective1Dense(obj.f_dense), len(obj.x_dense), obj.Theta,
      saga.Hook            {obj.Hook},
      saga.Gamma           {obj.stepSize},
      saga.Epsilon         {obj.Epsilon},
      saga.MaxIterations   {obj.MaxIterations},
      saga.Seed            {obj.Seed},
      saga.ProximalOperator{proxop},
      saga.JitUpdate       {jitUpdate}); err != nil {
      return err
    } else {
      obj.Seed = s
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
  if obj.sparse {
    for _, x := range obj.x_sparse {
      r  := 0.0
      it := x.ConstIterator()
      // skip first element
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
  } else {
    for _, x := range obj.x_dense {
      r  := 0.0
      it := x.ConstIterator()
      // skip first element
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
  }
  L := (0.25*(max_squared_sum + 1.0) + obj.L2Reg/float64(obj.n))
  L *= math.Max(obj.ClassWeights[0], obj.ClassWeights[1])
  obj.stepSize  = 1.0/(2.0*L + math.Min(2.0*obj.L2Reg, L))
  obj.stepSize *= obj.StepSizeFactor
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

/* -------------------------------------------------------------------------- */

type sagaLogisticRegressionL1state struct {
  x0              DenseBareRealVector
  x1              DenseBareRealVector
  xk            []int
  g1              saga.GradientJit
  g2              saga.GradientJit
  d               int
  n_x_old         int
  n_x_new         int
  l1_step         float64
  dict          []saga.GradientJit
  s               DenseBareRealVector
  xs            []bool
  ns              int
  cumulative_sums DenseBareRealVector
  t_n             BareReal
  t_g             BareReal
  jit             sagaJitUpdateL1
}

func (obj *LogisticRegression) sagaLogisticRegressionL1(
  f saga.Objective1Sparse,
  n int,
  x DenseBareRealVector,
  l1reg  saga.L1Regularization,
  autoReg saga.AutoReg,
  gamma saga.Gamma,
  epsilon saga.Epsilon,
  maxIterations saga.MaxIterations,
  hook saga.Hook,
  seed saga.Seed) (DenseBareRealVector, int64, error) {

  if len(obj.dict) == 0 || obj.d != x.Dim() {
    obj.x0 = AsDenseBareRealVector(x)
    obj.x1 = AsDenseBareRealVector(x)
    obj.xk = make([]int,  x.Dim())
    obj.xs = make([]bool, n)
    obj.ns = 0
    obj.cumulative_sums = NullDenseBareRealVector(n)

    // length of gradient
    obj.d = x.Dim()

    // some constants
    obj.t_n = BareReal(0.0)
    obj.t_g = BareReal(gamma.Value)

    // prevent that in auto-lambda mode the step size is initialized to zero
    if autoReg.Value > 0 && l1reg.Value == 0.0 {
      l1reg.Value = 1.0
    }
    obj.jit.SetLambda(l1reg.Value*gamma.Value/float64(n))

    // number of non-zero parameters used for auto-lambda mode
    obj.n_x_old = 0
    obj.n_x_new = 0
    // step size for auto-lambda mode
    obj.l1_step = 0.01*obj.jit.GetLambda()

    // sum of gradients
    obj.s = NullDenseBareRealVector(obj.d)
    // initialize s and d
    obj.dict = make([]saga.GradientJit, n)
    for i := 0; i < n; i++ {
      if _, w, g, err := f(i, nil); err != nil {
        return nil, seed.Value, err
      } else {
        obj.dict[i].Set(w, g)
      }
    }
  }
  g := rand.New(rand.NewSource(seed.Value))

  for epoch := 0; epoch < maxIterations.Value; epoch++ {
    for i_ := 0; i_ < n; i_++ {
      j := i_
      if seed.Value != -1 {
        j = g.Intn(n)
      }
      if !obj.xs[j] {
        obj.ns += 1
        obj.t_n = BareReal(obj.ns)
      }
      if i_ == 0 {
        obj.cumulative_sums[0 ] = obj.t_g/obj.t_n
      } else {
        obj.cumulative_sums[i_] = obj.cumulative_sums[i_-1] + obj.t_g/obj.t_n
      }
      // get old gradient
      obj.g1 = obj.dict[j]
      // perform jit updates for all x_i where g_i != 0
      for _, k := range obj.g1.G.GetSparseIndices() {
        if m := i_-obj.xk[k]; m > 0 {
          cum_sum := obj.cumulative_sums[i_-1]
          if obj.xk[k] != 0 {
            cum_sum -= obj.cumulative_sums[obj.xk[k]-1]
          }
          obj.x1[k] = obj.jit.Update(obj.x1[k], cum_sum*obj.s[k]/BareReal(m), k, m)
        }
      }
      // evaluate objective function
      if _, w, gt, err := f(j, obj.x1); err != nil {
        return obj.x1, g.Int63(), err
      } else {
        obj.g2.Set(w, gt)
      }
      c := BareReal(obj.g2.W - obj.g1.W)
      v := obj.g1.G.GetSparseValues()
      for i, k := range obj.g1.G.GetSparseIndices() {
        obj.x1[k] = obj.x1[k] - obj.t_g*(1.0 - 1.0/obj.t_n)*c*BareReal(v[i])
        obj.xk[k] = i_
      }
      if !obj.xs[j] {
        obj.xs[j] = true
        obj.g2.Add(obj.s)
      } else {
        // update gradient avarage
        obj.g1.Update(obj.g2, obj.s)
      }
      // update dictionary
      obj.dict[j].Set(obj.g2.W, obj.g2.G)
    }
    // compute missing updates of x1
    for k := 0; k < obj.x1.Dim(); k++ {
      if m := n-obj.xk[k]; m > 0 {
        cum_sum := obj.cumulative_sums[n-1]
        if obj.xk[k] != 0 {
          cum_sum -= obj.cumulative_sums[obj.xk[k]-1]
        }
        obj.x1[k] = obj.jit.Update(obj.x1[k], cum_sum*obj.s[k]/BareReal(m), k, m)
      }
      // reset xk
      obj.xk[k] = 0
    }
    if stop, delta, err := saga.EvalStopping(obj.x0, obj.x1, epsilon.Value*gamma.Value); stop {
      return obj.x1, g.Int63(), err
    } else {
      // execute hook if available
      if hook.Value != nil && hook.Value(obj.x1, ConstReal(delta), epoch) {
        break
      }
    }
    // update lambda
    if autoReg.Value > 0 {
      obj.n_x_new = 0
      // count number of non-zero entries
      for k := 1; k < obj.x1.Dim(); k++ {
        if obj.x1[k] != 0.0 {
          obj.n_x_new += 1
        }
      }
      switch {
      case obj.n_x_old < autoReg.Value && obj.n_x_new < autoReg.Value: fallthrough
      case obj.n_x_old > autoReg.Value && obj.n_x_new > autoReg.Value:
        obj.l1_step = 1.2*obj.l1_step
      default:
        obj.l1_step = 0.8*obj.l1_step
      }
      if obj.n_x_new < autoReg.Value {
        obj.jit.SetLambda(obj.jit.GetLambda() - obj.l1_step)
      } else
      if obj.n_x_new > autoReg.Value {
        obj.jit.SetLambda(obj.jit.GetLambda() + obj.l1_step)
      }
      if obj.jit.GetLambda() < 0.0 {
        obj.jit.SetLambda(0.0)
      }
      // swap old and new counts
      obj.n_x_old, obj.n_x_new = obj.n_x_new, obj.n_x_old
    }
    obj.x0.SET(obj.x1)
  }
  return obj.x1, g.Int63(), nil
}
