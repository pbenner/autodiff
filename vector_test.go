/* Copyright (C) 2015 Philipp Benner
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

package autodiff

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"
import "testing"

/* -------------------------------------------------------------------------- */

func TestVector(t *testing.T) {

  v := NewVector(RealType, []float64{1,2,3,4,5,6})

  if v[1].GetValue() != 2.0 {
    t.Error("Vector initialization failed!")
  }
}

func TestVectorSort(t *testing.T) {

  v1 := NewVector(RealType, []float64{4,3,7,4,1,29,6})
  v2 := NewVector(RealType, []float64{4,3,7,4,1,29,6})

  v1.Sort(false)
  v2.Sort(true)

  if v1[6].GetValue() != 29.0 {
    t.Error("Vector sorting failed!")
  }
  if v2[6].GetValue() != 1.0 {
    t.Error("Vector sorting failed!")
  }
}

func TestVectorToMatrix(t *testing.T) {

  v := NewVector(RealType, []float64{1,2,3,4,5,6})
  m := v.Matrix(2, 3)

  if m.At(1,0).GetValue() != 4 {
    t.Error("Vector to matrix conversion failed!")
  }
}

func TestVdotV(t *testing.T) {

  a := NewVector(RealType, []float64{1, 2,3,4})
  b := NewVector(RealType, []float64{2,-1,1,7})
  r := VdotV(a, b)

  if r.GetValue() != 31 {
    t.Error("VmulV() failed!")
  }
}

func TestVmulV(t *testing.T) {

  a := NewVector(RealType, []float64{1, 2,3,4})
  b := NewVector(RealType, []float64{2,-1,1,7})
  r := VmulV(a, b)

  if r[1].GetValue() != -2 {
    t.Error("VmulV() failed!")
  }
}

func TestReadVector(t *testing.T) {

  v, err := ReadVector(RealType, "vector_test.table")
  if err != nil {
    panic(err)
  }
  r := NewVector(RealType, []float64{1,2,3,4,5,6})

  if Vnorm(VsubV(v, r)).GetValue() != 0.0 {
    t.Error("Read vector failed!")
  }
}

func TestVectorMapReduce(t *testing.T) {

  r1 := NewVector(RealType, []float64{2.718282e+00, 7.389056e+00, 2.008554e+01, 5.459815e+01})
  r2 := 84.79103
  a := NewVector(RealType, []float64{1, 2,3,4})
  a.Map(Exp)
  b := a.Reduce(Add)

  if Vnorm(VsubV(a,r1)).GetValue() > 1e-2 {
    t.Error("Vector map/reduce failed!")
  }
  if math.Abs(b.GetValue() - r2) > 1e-2 {
    t.Error("Vector map/reduce failed!")
  }
}
