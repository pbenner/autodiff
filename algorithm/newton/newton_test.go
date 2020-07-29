/* Copyright (C) 2015-2020 Philipp Benner
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

package newton

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestNewtonRoot(test *testing.T) {

  t := NewFloat64(0.0)
  f := func(x ConstVector) (MagicVector, error) {
    y  := NullDenseReal64Vector(2)
    t1 := NullReal64()
    t2 := NullReal64()
    // y1 = x1^2 + x2^2 - 6
    t1.Pow(x.ConstAt(0), ConstFloat64(2))
    t2.Pow(x.ConstAt(1), ConstFloat64(2))
    y.At(0).Sub(t1.Add(t1, t2), ConstFloat64(6))
    // y2 = x1^3 - x2^2
    t1.Pow(x.ConstAt(0), ConstFloat64(3))
    t2.Pow(x.ConstAt(1), ConstFloat64(2))
    y.At(1).Sub(t1, t2)

    return y, nil
  }
  v1 := NewDenseFloat64Vector([]float64{1, 1})
  v2 := NewDenseFloat64Vector([]float64{1.537656, 1.906728})
  v3, err := RunRoot(f, v1, Epsilon{1e-8})
  if err != nil {
    test.Error(err)
  } else {
    if t.Vnorm(v2.VsubV(v2, v3)).GetFloat64() > 1e-6  {
      test.Error("Newton method failed!")
    }
  }
}

func TestNewtonCrit1(test *testing.T) {
  t := NewFloat64(0.0)
  f := func(x ConstVector) (MagicScalar, error) {
    // minimize x^2 subject to x^2 = 1, which is equivalent to finding
    // the critical points of the lagrangian x^2 + lambda(x^2 - 1)
    t1 := NullReal64()
    t2 := NullReal64()
    t1.Mul(x.ConstAt(0), x.ConstAt(0))
    t2.Mul(x.ConstAt(0), x.ConstAt(0))
    t1.Add(t2, t1.Mul(x.ConstAt(1), t1.Sub(t1, ConstFloat64(1))))

    return t1, nil
  }
  v1 := NewDenseFloat64Vector([]float64{3,  5})
  v2 := NewDenseFloat64Vector([]float64{1, -1})
  v3, err := RunCrit(f, v1, Epsilon{1e-8})
  if err != nil {
    test.Error(err)
  } else {
    if t.Vnorm(v2.VsubV(v2, v3)).GetFloat64() > 1e-6  {
      test.Error("Newton method failed!")
    }
  }
}

func TestNewtonCrit2(test *testing.T) {
  t := NewFloat64(0.0)
  // define Lagrangian function
  f := func(x ConstVector) (MagicScalar, error) {
    // minimize x + y subject to x^2 + y^2 = 1, which is equivalent to finding
    // the critical points of the lagrangian x^2 + lambda(x^2 - 1)
    t1 := NullReal64()
    t2 := NullReal64()
    t3 := NullReal64()
    t1.Mul(x.ConstAt(1), x.ConstAt(1))
    t2.Mul(x.ConstAt(0), x.ConstAt(0))
    t3.Add(x.ConstAt(0), x.ConstAt(1))
    t3.Add(t3, t1.Mul(x.ConstAt(2), t1.Sub(t1.Add(t2, t1), ConstFloat64(1))))

    return t3, nil
  }
  v1 := NewDenseFloat64Vector([]float64{3,  5, 1})
  v2 := NewDenseFloat64Vector([]float64{math.Sqrt(2.0)/2.0, math.Sqrt(2.0)/2.0, -math.Sqrt(2.0)/2.0})
  v3, err := RunCrit(f, v1, Epsilon{1e-8})
  if err != nil {
    test.Error(err)
  } else {
    if t.Vnorm(v2.VsubV(v2, v3)).GetFloat64() > 1e-6  {
      test.Error("Newton method failed!")
    }
  }
}
