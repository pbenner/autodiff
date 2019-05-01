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

import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type LogisticRegression struct {
  *vectorDistribution.LogisticRegression
  sparse     bool
  n          int
  m          int
  x_sparse []*SparseBareRealVector
  x_dense  []  DenseBareRealVector
  x        []ConstVector
  c        []bool
}

/* -------------------------------------------------------------------------- */

func NewLogisticRegression(index []int, theta_ []float64, n int) (*LogisticRegression, error) {
  r := LogisticRegression{}
  if index == nil {
    r.sparse = false
    if dist, err := vectorDistribution.NewLogisticRegression(NewDenseBareRealVector(theta_)); err != nil {
      return nil, err
    } else {
      r.LogisticRegression = dist
    }
  } else {
    r.sparse = true
    if dist, err := vectorDistribution.NewLogisticRegression(NewSparseBareRealVector(index, theta_, n)); err != nil {
      return nil, err
    } else {
      r.LogisticRegression = dist
    }
  }
  return nil, nil
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) Clone() *LogisticRegression {
  r := LogisticRegression{}
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

// x_i = (class_label, x_i1, x_i2, ..., x_im)
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
  if x[0].Dim() <= 1 {
    return fmt.Errorf("vector has invalid dimension")
  }
  if obj.sparse {
    for i, _ := range x {
      if x[i].Dim() != x[0].Dim() {
        return fmt.Errorf("data has inconsistent dimensions")
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
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *LogisticRegression) Estimate(gamma ConstVector, p ThreadPool) error {
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
