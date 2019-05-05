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

import   "github.com/pbenner/autodiff/algorithm/saga"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type LogisticRegression struct {
  *vectorDistribution.LogisticRegression
  sparse     bool
  n          int
  x_sparse []*SparseBareRealVector
  x_dense  []  DenseBareRealVector
  x        []ConstVector
  c        []bool
  stepSize   float64
  // optional parameters
  Epsilon    float64
  L1Reg      float64
  L2Reg      float64
  Hook       func(x ConstVector, step, y ConstScalar, i int) bool
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
    if dist, err := vectorDistribution.NewLogisticRegression(NewDenseBareRealVector(theta)); err != nil {
      return nil, err
    } else {
      r.LogisticRegression = dist
    }
  } else {
    r.sparse = true
    if dist, err := vectorDistribution.NewLogisticRegression(NewSparseBareRealVector(index, theta, n)); err != nil {
      return nil, err
    } else {
      r.LogisticRegression = dist
    }
  }
  return &r, nil
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) Clone() *LogisticRegression {
  r := LogisticRegression{}
  // copy data and optional arguments
  r  = *obj
  r.LogisticRegression = obj.LogisticRegression.Clone()
  return &r
}

func (obj *LogisticRegression) CloneVectorEstimator() VectorEstimator {
  return obj.Clone()
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
  if k := obj.LogisticRegression.Dim()+2; x[0].Dim() != k {
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
      case *SparseBareRealVector:
        obj.x_sparse = append(obj.x_sparse, a)
      default:
        obj.x_sparse = append(obj.x_sparse, AsSparseBareRealVector(t))
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
      case DenseBareRealVector:
        obj.x_dense = append(obj.x_dense, a)
      default:
        obj.x_dense = append(obj.x_dense, AsDenseBareRealVector(t))
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
  if obj.sparse {
    theta := obj.LogisticRegression.GetParameters()
    prox  := saga.ProximalOperatorSparse(nil)
    switch {
    case obj.L1Reg != 0.0: prox = proxL1sparse(obj.stepSize*obj.L1Reg)
    case obj.L2Reg != 0.0: prox = proxTIsparse(obj.stepSize*obj.L2Reg)
    }
    if r, err := saga.Run(saga.ObjectiveSparse(obj.f_sparse), len(obj.x_sparse), theta,
      saga.Hook   {obj.Hook},
      saga.Gamma  {obj.stepSize},
      saga.Epsilon{obj.Epsilon},
      prox); err != nil {
      return err
    } else {
      obj.LogisticRegression.SetParameters(r)
    }
  } else {
    theta := obj.LogisticRegression.GetParameters()
    prox  := saga.ProximalOperatorDense(nil)
    switch {
    case obj.L1Reg != 0.0: prox = proxL1dense(obj.stepSize*obj.L1Reg)
    case obj.L2Reg != 0.0: prox = proxTIdense(obj.stepSize*obj.L2Reg)
    }
    if r, err := saga.Run(saga.ObjectiveDense(obj.f_dense), len(obj.x_dense), theta,
      saga.Hook   {obj.Hook},
      saga.Gamma  {obj.stepSize},
      saga.Epsilon{obj.Epsilon},
      prox); err != nil {
      return err
    } else {
      obj.LogisticRegression.SetParameters(r)
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
  return obj.LogisticRegression, nil
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) setStepSize() {
  max_squared_sum := 0.0
  for i, _ := range obj.x {
    r := 0.0
    for it := obj.x[i].ConstIterator(); it.Ok(); it.Next() {
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

func proxL1dense(lambda float64) saga.ProximalOperatorDense {
  g := saga.ProxL1Dense(lambda)
  f := func(x, w DenseBareRealVector, t *BareReal) {
    g(x, w, t)
    // do not regularize intercept
    x.AT(0).SET(w.AT(0))
  }
  return f
}

func proxTIdense(lambda float64) saga.ProximalOperatorDense {
  g := saga.ProxTiDense(lambda)
  f := func(x, w DenseBareRealVector, t *BareReal) {
    g(x, w, t)
    // do not regularize intercept
    x.AT(0).SET(w.AT(0))
  }
  return f
}

func proxL1sparse(lambda float64) saga.ProximalOperatorSparse {
  g := saga.ProxL1Sparse(lambda)
  f := func(x, w *SparseBareRealVector, t *BareReal) {
    g(x, w, t)
    // do not regularize intercept
    x.AT(0).SET(w.AT(0))
  }
  return f
}

func proxTIsparse(lambda float64) saga.ProximalOperatorSparse {
  g := saga.ProxTiSparse(lambda)
  f := func(x, w *SparseBareRealVector, t *BareReal) {
    g(x, w, t)
    // do not regularize intercept
    x.AT(0).SET(w.AT(0))
  }
  return f
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) f_dense(i int, theta DenseBareRealVector) (ConstReal, ConstReal, DenseBareRealVector, bool, error) {
  r := BareReal(0.0)
  x := obj.x_dense
  y := ConstReal(0.0)
  w := ConstReal(0.0)
  if i >= len(x) {
    return y, w, nil, true, fmt.Errorf("index out of bounds")
  }
  if err := obj.LogisticRegression.SetParameters(theta); err != nil {
    return y, w, nil, true, err
  }
  if err := obj.LogisticRegression.LogPdf(&r, x[i]); err != nil {
    return y, w, nil, true, err
  }
  if math.IsNaN(r.GetValue()) {
    return y, w, nil, true, fmt.Errorf("NaN value detected")
  }
  y = ConstReal(r.GetValue())
  if obj.c[i] {
    w = ConstReal(math.Exp(r.GetValue()) - 1.0)
  } else {
    w = ConstReal(math.Exp(r.GetValue()))
  }
  return y, w, x[i], true, nil
}

func (obj *LogisticRegression) f_sparse(i int, theta *SparseBareRealVector) (ConstReal, ConstReal, *SparseBareRealVector, bool, error) {
  r := BareReal(0.0)
  x := obj.x_sparse
  y := ConstReal(0.0)
  w := ConstReal(0.0)
  if i >= len(x) {
    return y, w, nil, true, fmt.Errorf("index out of bounds")
  }
  if err := obj.LogisticRegression.SetParameters(theta); err != nil {
    return y, w, nil, true, err
  }
  if err := obj.LogisticRegression.LogPdf(&r, x[i]); err != nil {
    return y, w, nil, true, err
  }
  if math.IsNaN(r.GetValue()) {
    return y, w, nil, true, fmt.Errorf("NaN value detected")
  }
  y = ConstReal(r.GetValue())
  if obj.c[i] {
    w = ConstReal(math.Exp(r.GetValue()) - 1.0)
  } else {
    w = ConstReal(math.Exp(r.GetValue()))
  }
  return y, w, x[i], true, nil
}
