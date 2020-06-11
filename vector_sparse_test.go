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

package autodiff

/* -------------------------------------------------------------------------- */

//import "fmt"
import "encoding/json"
import "testing"

/* -------------------------------------------------------------------------- */

func TestSparseVector1(test *testing.T) {

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
	w := NewSparseRealVector([]int{1, 210, 310, 30, 10192}, []float64{1, 3, 4, -5, 6}, 20000)
	r := NewSparseRealVector([]int{1, 210, 310, 30, 10192}, []float64{1, 9, 16, 25, 36}, 20000)
	t := NullReal()

	v.VmulV(v, w)

	if t.Vnorm(r.VsubV(r, v)); t.GetValue() > 0.0 {
		test.Errorf("test failed")
	}
}

func TestSparseVector1Const(test *testing.T) {

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
	w := NewSparseConstRealVector([]int{1, 210, 310, 30, 10192}, []float64{1, 3, 4, -5, 6}, 20000)
	r := NewSparseRealVector([]int{1, 210, 310, 30, 10192}, []float64{1, 9, 16, 25, 36}, 20000)
	t := NullReal()

	v.VmulV(v, w)

	if t.Vnorm(r.VsubV(r, v)); t.GetValue() > 0.0 {
		test.Errorf("test failed")
	}
}

func TestSparseVector2(test *testing.T) {

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
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

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{10, -2, 3, 4, -5, 6}, 20000)
	w1 := NewSparseRealVector([]int{19999, 1, 19996, 19997, 0, 19998}, []float64{10, -2, 3, 4, -5, 6}, 20000)
	w2 := NewSparseRealVector([]int{0, 19998, 3, 2, 19999, 1}, []float64{10, -2, 3, 4, -5, 6}, 20000)
	t := NullReal()

	v.Sort(false)
	if t.Vnorm(w1.VsubV(w1, v)); t.GetValue() > 0.0 {
		test.Errorf("test failed")
	}

	v.Sort(true)
	if t.Vnorm(w2.VsubV(w2, v)); t.GetValue() > 0.0 {
		test.Errorf("test failed")
	}
}

func TestSparseVector4(test *testing.T) {

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
	w := v.AppendScalar(NewReal(30))
	r := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192, 20000}, []float64{1, 2, 3, 4, -5, 6, 30}, 20001)
	t := NullReal()

	if t.Vnorm(w.VsubV(w, r)); t.GetValue() > 0.0 {
		test.Errorf("test failed")
	}
}

func TestSparseVector4Const(test *testing.T) {

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
	w := v.AppendScalar(NewReal(30))
	r := NewSparseConstRealVector([]int{1, 100, 210, 310, 30, 10192, 20000}, []float64{1, 2, 3, 4, -5, 6, 30}, 20001)
	t := NullReal()

	if t.Vnorm(w.VsubV(w, r)); t.GetValue() > 0.0 {
		test.Errorf("test failed")
	}
}

func TestSparseVector5(test *testing.T) {

	i := 101
	j := 400

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
	w := v.Slice(i, j)
	r := NewSparseRealVector([]int{210 - i, 310 - i}, []float64{3, 4}, j-i)
	t := NullReal()

	if t.Vnorm(r.VsubV(r, w)); t.GetValue() > 0.0 {
		test.Errorf("test failed")
	}
}

func TestSparseVector5Const(test *testing.T) {

	i := 101
	j := 400

	v := NewSparseConstRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
	w := v.ConstSlice(i, j)
	r := NewSparseRealVector([]int{210 - i, 310 - i}, []float64{3, 4}, j-i)
	t := NullReal()

	if t.Vnorm(r.VsubV(r, w)); t.GetValue() > 0.0 {
		test.Errorf("test failed")
	}
}

func TestSparseVector6(test *testing.T) {

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
	r := []float64{1.0, -5.0, 2.0, 3.0, 4.0, 3.785, 6.0}

	i := 0
	for it := v.ITERATOR(); it.Ok(); it.Next() {
		// insert new value
		if i == 2 {
			v.AT(1000).SetValue(3.785)
		}
		if s := it.GET(); s.GetValue() != r[i] {
			test.Error("test failed")
		}
		i++
	}
	if i != len(r) {
		test.Error("test failed")
	}
}

func TestSparseVector7(test *testing.T) {

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
	r := []float64{1.0, -5.0, 2.0, 3.0, 6.0}

	i := 0
	for it := v.ITERATOR(); it.Ok(); it.Next() {
		// insert new value
		if i == 2 {
			v.AT(310).SetValue(0.0)
			// apply changes
			for is := v.ITERATOR(); is.Ok(); is.Next() {
			}
		}
		if s := it.GET(); i >= len(r) || s.GetValue() != r[i] {
			test.Error("test failed")
		}
		i++
	}
	if i != len(r) {
		test.Error("test failed")
	}
}

func TestSparseVector8(test *testing.T) {

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
	r := []float64{1.0, -5.0, 2.0, 110.0, 3.0, 4.0, 6.0}

	i := 0
	for it := v.ITERATOR(); it.Ok(); it.Next() {
		if s := it.GET(); i >= len(r) || s.GetValue() != r[i] {
			test.Error("test failed")
		}
		// insert new value
		if i == 2 {
			v.AT(100).SetValue(0.0)
			// apply changes
			for is := v.ITERATOR(); is.Ok(); is.Next() {
			}
			v.AT(110).SetValue(110.0)
		}
		i++
	}
	if i != len(r) {
		test.Error("test failed")
	}
}

func TestSparseVector9(test *testing.T) {

	v := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 2, 3, 4, -5, 6}, 20000)
	r := NewSparseRealVector([]int{1, 100, 210, 310, 30, 10192}, []float64{1, 3, 2, 4, -5, 6}, 20000)
	t := NullReal()

	v.Swap(100, 210)
	if t.Vnorm(r.VsubV(r, v)); t.GetValue() > 0 {
		test.Errorf("test failed")
	}
}
