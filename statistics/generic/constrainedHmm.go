/* Copyright (C) 2018 Philipp Benner
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

package generic

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/newton"

/* -------------------------------------------------------------------------- */

type ChmmTransitionMatrix struct {
  Matrix
  Tree HmmNode
}

func NewChmmTransitionMatrix(tr Matrix, tree HmmNode, isLog bool) (ChmmTransitionMatrix, error) {
  tr = tr.CloneMatrix()
  // log-transform all probabilities
  if !isLog {
    tr.Map(func(x Scalar) { x.Log(x) })
  }
  r := ChmmTransitionMatrix{tr, tree}
  if err := r.Normalize(); err != nil {
    return ChmmTransitionMatrix{}, err
  }
  return r, nil
}

func (obj ChmmTransitionMatrix) GetMatrix() Matrix {
  return obj.Matrix
}

func (obj ChmmTransitionMatrix) Normalize() error {
  if lambda, err := obj.computeLambda(); err != nil {
    return err
  } else {
    obj.normalize(obj.Tree, lambda)
    return nil
  }
}

func (obj ChmmTransitionMatrix) CloneTransitionMatrix() TransitionMatrix {
  return ChmmTransitionMatrix{
    obj.Matrix.CloneMatrix(),
    obj.Tree }
}

/* -------------------------------------------------------------------------- */

func (obj ChmmTransitionMatrix) computeLambda() (Vector, error) {
  tr   := obj.Matrix
  n, _ := tr.Dims()
  // lambda
  l := NullVector(RealType, n)
  x := NullVector(RealType, n)
  // objective function
  f := func(lambda Vector) (Vector, error) {
    obj.EvalConstraints(lambda, x)
    return x, nil
  }
  if r, err := newton.RunRoot(f, l, newton.Epsilon{1e-8}); err != nil {
    return nil, err
  } else {
    return r, nil
  }
}

/* -------------------------------------------------------------------------- */

func (obj ChmmTransitionMatrix) normalizeLeaf(node HmmNode, lambda Vector) {
  tr := obj.Matrix
  // row/column ranges
  from := node.States[0]
  to   := node.States[1]
  for i := from; i < to; i++ {
    for j := from; j < to; j++ {
      tr.At(i,j).Sub(tr.At(i,j), lambda.At(i))
    }
  }
}

func (obj ChmmTransitionMatrix) normalizeInt(rfrom, rto, cfrom, cto int, lambda Vector) {
  tr := obj.Matrix
  t  := tr.ElementType()
  // n = (n_i) the number of non-zero entries in row i
  n  := make([]int, rto-rfrom)
  // a' = sum xi_i
  ap := NewScalar(t, math.Inf(-1))
  // z' = sum n_i lambda_i
  zp := NewScalar(t, math.Inf(-1))
  t1 := NewScalar(t, math.Inf(-1))
  t2 := NewScalar(t, math.Inf(-1))
  for i := rfrom; i < rto; i++ {
    for j := cfrom; j < cto; j++ {
      ap.LogAdd(ap, tr.At(i, j), t2)
    }
  }
  // count number of non-zero entries for each row i
  for i := rfrom; i < rto; i++ {
    for j := cfrom; j < cto; j++ {
      if !math.IsInf(tr.At(i,j).GetValue(), -1) {
        n[i-rfrom]++
      }
    }
  }
  // compute z'
  for i := rfrom; i < rto; i++ {
    if n[i-rfrom] != 0 {
      t1.Add(ConstReal(math.Log(float64(n[i-rfrom]))), lambda.At(i))
      zp.LogAdd(zp, t1, t2)
    }
  }
  // compute a
  ap.Sub(ap, zp)
  // normalize matrix
  for i := rfrom; i < rto; i++ {
    for j := cfrom; j < cto; j++ {
      if !math.IsInf(tr.At(i,j).GetValue(), -1) {
        tr.At(i,j).Set(ap)
      }
    }
  }
}

func (obj ChmmTransitionMatrix) normalize(node HmmNode, lambda Vector) {
  if n := len(node.Children); n == 0 {
    // this is a leaf node
    obj.normalizeLeaf(node, lambda)
  } else {
    // this is an internal node
    for i := 0; i < n; i++ {
      // normalize transitions between child i and all other
      // children
      rfrom := node.Children[i].States[0]
      rto   := node.Children[i].States[1]
      for j := 0; j < n; j++ {
        if i == j {
          obj.normalize(node.Children[i], lambda)
        } else {
          cfrom := node.Children[j].States[0]
          cto   := node.Children[j].States[1]
          obj.normalizeInt(rfrom, rto, cfrom, cto, lambda)
        }
      }
    }
  }
}

/* evaluate transition matrix constraints given lagrangian multipliers
 * -------------------------------------------------------------------------- */

func (obj ChmmTransitionMatrix) evalConstraintsLeaf(node HmmNode, lambda ConstVector, x Vector) {
  tr := obj.Matrix
  t  := x.ElementType()
  t1 := NewScalar(t, math.Inf(-1))
  t2 := NewScalar(t, math.Inf(-1))
  // row/column ranges
  from := node.States[0]
  to   := node.States[1]
  for i := from; i < to; i++ {
    for j := from; j < to; j++ {
      t1.Sub(tr.At(i,j), lambda.ConstAt(i))
      x.At(i).LogAdd(x.At(i), t1, t2)
    }
  }
}

func (obj ChmmTransitionMatrix) evalConstraintsInt(rfrom, rto, cfrom, cto int, lambda ConstVector, x Vector) {
  tr := obj.Matrix
  t  := x.ElementType()
  // n = (n_i) the number of non-zero entries in row i
  n  := make([]int, rto-rfrom)
  // a' = sum xi_i
  ap := NewScalar(t, math.Inf(-1))
  // z' = sum n_i lambda_i
  zp := NewScalar(t, math.Inf(-1))
  t1 := NewScalar(t, math.Inf(-1))
  t2 := NewScalar(t, math.Inf(-1))
  for i := rfrom; i < rto; i++ {
    for j := cfrom; j < cto; j++ {
      ap.LogAdd(ap, tr.At(i, j), t2)
    }
  }
  // count number of non-zero entries for each row i
  for i := rfrom; i < rto; i++ {
    for j := cfrom; j < cto; j++ {
      if !math.IsInf(tr.At(i,j).GetValue(), -1) {
        n[i-rfrom]++
      }
    }
  }
  // compute z'
  for i := rfrom; i < rto; i++ {
    if n[i-rfrom] != 0 {
      t1.Add(ConstReal(math.Log(float64(n[i-rfrom]))), lambda.ConstAt(i))
      zp.LogAdd(zp, t1, t2)
    }
  }
  // compute a = a'/z' and x_i += n_i a_i
  for i := rfrom; i < rto; i++ {
    if n[i-rfrom] != 0 {
      t1.Sub(ap, zp)
      t1.Add(ConstReal(math.Log(float64(n[i-rfrom]))), t1)
      x.At(i).LogAdd(x.At(i), t1, t2)
    }
  }
}

func (obj ChmmTransitionMatrix) evalConstraints(node HmmNode, lambda ConstVector, x Vector) {
  if n := len(node.Children); n == 0 {
    // this is a leaf node
    obj.evalConstraintsLeaf(node, lambda, x)
  } else {
    // this is an internal node
    for i := 0; i < n; i++ {
      // normalize transitions between child i and all other
      // children
      rfrom := node.Children[i].States[0]
      rto   := node.Children[i].States[1]
      for j := 0; j < n; j++ {
        if i == j {
          obj.evalConstraints(node.Children[i], lambda, x)
        } else {
          cfrom := node.Children[j].States[0]
          cto   := node.Children[j].States[1]
          obj.evalConstraintsInt(rfrom, rto, cfrom, cto, lambda, x)
        }
      }
    }
  }
}

func (obj ChmmTransitionMatrix) EvalConstraints(lambda ConstVector, x Vector) error {
  tr   := obj.Matrix
  n, m := tr.Dims()
  if n != m {
    return fmt.Errorf("tr is not a square-matrix")
  }
  if n != lambda.Dim() {
    return fmt.Errorf("lambda has invalid dimension")
  }
  if n != x.Dim() {
    return fmt.Errorf("x has invalid dimension")
  }
  x.Map(func(xi Scalar) { xi.SetValue(math.Inf(-1)) })

  obj.evalConstraints(obj.Tree, lambda, x)

  x.Map(func(xi Scalar) {
    xi.Exp(xi)
    xi.Sub(xi, ConstReal(1.0))
  })

  return nil
}
