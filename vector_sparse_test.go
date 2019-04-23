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
import "encoding/json"
import "testing"

/* -------------------------------------------------------------------------- */

func TestSparseVector1(test *testing.T) {

  v := NewSparseRealVector([]int{1,100,210,310,30,10192}, []float64{1,2,3, 4,-5, 6}, 20000)
  w := NewSparseRealVector([]int{1,    210,310,30,10192}, []float64{1,  3, 4,-5, 6}, 20000)
  r := NewSparseRealVector([]int{1,    210,310,30,10192}, []float64{1,  9,16,25,36}, 20000)
  t := NullReal()

  v.VmulV(v, w)

  if t.Vnorm(r.VsubV(r, v)); t.GetValue() > 0.0 {
    test.Errorf("test failed")
  }
}

func TestSparseVector2(test *testing.T) {

  v := NewSparseRealVector([]int{1,100,210,310,30,10192}, []float64{1,2,3,4,-5,6}, 20000)
  w := NewSparseRealVector(nil, nil, 0)
  t := NullReal()

  if v_bytes, err := json.Marshal(v); err != nil {
    test.Errorf("test failed")
  } else {
    if err := json.Unmarshal(v_bytes, w); err != nil {
      test.Errorf("test failed")
    } else {
      if t.Vnorm(v.VsubV(v, w)); t.GetValue() > 0.0 {
        test.Errorf("test failed")
      }
    }
  }
}

func TestSparseVector3(test *testing.T) {

  v  := NewSparseRealVector([]int{    1,  100,    210,   310,    30, 10192}, []float64{10, -2, 3,  4, -5, 6}, 20000)
  w1 := NewSparseRealVector([]int{19999,     1, 19996, 19997,     0, 19998}, []float64{10, -2, 3,  4, -5, 6}, 20000)
  w2 := NewSparseRealVector([]int{    0, 19998,     3,     2, 19999,     1}, []float64{10, -2, 3,  4, -5, 6}, 20000)
  t  := NullReal()

  v.Sort(false)  
  if t.Vnorm(w1.VsubV(w1, v)); t.GetValue() > 0.0 {
    test.Errorf("test failed")
  }

  v.Sort(true)
  if t.Vnorm(w2.VsubV(w2, v)); t.GetValue() > 0.0 {
    test.Errorf("test failed")
  }
}
