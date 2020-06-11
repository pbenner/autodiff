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

import "fmt"
import "encoding/json"
import "math"

/* -------------------------------------------------------------------------- */

// This is the basic state used by real and probability scalars.
type BasicState struct {
	Value      float64
	Order      int
	Derivative []float64
	Hessian    [][]float64
	N          int
}

/* constructors
 * -------------------------------------------------------------------------- */

// Create a new basic state. As an optional argument the number of variables
// for which derivatives are computed may be passed.
func NewBasicState(value float64) *BasicState {
	a := BasicState{}
	a.Value = value
	a.Order = 0
	a.N = 0
	return &a
}

/* -------------------------------------------------------------------------- */

// Allocate memory for derivatives of n variables.
func (a *BasicState) Alloc(n, order int) {
	if a.N != n || a.Order != order {
		a.N = n
		a.Order = order
		// allocate gradient if requested
		if a.Order >= 1 {
			a.Derivative = make([]float64, n)
			// allocate Hessian if requested
			if a.Order >= 2 {
				a.Hessian = make([][]float64, n)
				for i := 0; i < n; i++ {
					a.Hessian[i] = make([]float64, n)
				}
			} else {
				a.Hessian = nil
			}
		} else {
			a.Derivative = nil
		}
	}
}

// Allocate memory for the results of mathematical operations on
// the given variables.
func (c *BasicState) AllocForOne(a ConstScalar) {
	c.Alloc(a.GetN(), a.GetOrder())
}
func (c *BasicState) AllocForTwo(a, b ConstScalar) {
	c.Alloc(iMax(a.GetN(), b.GetN()), iMax(a.GetOrder(), b.GetOrder()))
}

/* read access
 * -------------------------------------------------------------------------- */

// Indicates the maximal order of derivatives that are computed for this
// variable. `0' means no derivatives, `1' only the first derivative, and
// `2' the first and second derivative.
func (a *BasicState) GetOrder() int {
	return a.Order
}

// Returns the value of the variable.
func (a *BasicState) GetValue() float64 {
	return a.Value
}

// Returns the value of the variable on log scale.
func (a *BasicState) GetLogValue() float64 {
	return math.Log(a.Value)
}

// Returns the derivative of the ith variable.
func (a *BasicState) GetDerivative(i int) float64 {
	if a.Order >= 1 {
		return a.Derivative[i]
	} else {
		return 0.0
	}
}

func (a *BasicState) GetHessian(i, j int) float64 {
	if a.Order >= 2 {
		return a.Hessian[i][j]
	} else {
		return 0.0
	}
}

// Number of variables for which derivates are stored.
func (a *BasicState) GetN() int {
	return a.N
}

/* write access
 * -------------------------------------------------------------------------- */

func (a *BasicState) Reset() {
	a.Value = 0.0
	a.ResetDerivatives()
}

func (a *BasicState) ResetDerivatives() {
	if a.Order >= 1 {
		for i := 0; i < a.N; i++ {
			a.Derivative[i] = 0.0
		}
		if a.Order >= 2 {
			for i := 0; i < a.N; i++ {
				for j := 0; j < a.N; j++ {
					a.Hessian[i][j] = 0.0
				}
			}
		}
	}
}

// Set the state to b. This includes the value and all derivatives.
func (a *BasicState) Set(b ConstScalar) {
	a.Value = b.GetValue()
	a.Order = b.GetOrder()
	a.Alloc(b.GetN(), b.GetOrder())
	if a.Order >= 1 {
		for i := 0; i < b.GetN(); i++ {
			a.Derivative[i] = b.GetDerivative(i)
		}
		if a.Order >= 2 {
			for i := 0; i < b.GetN(); i++ {
				for j := 0; j < b.GetN(); j++ {
					a.Hessian[i][j] = b.GetHessian(i, j)
				}
			}
		}
	}
}

func (a *BasicState) SET(b *BasicState) {
	a.Value = b.GetValue()
	a.Order = b.GetOrder()
	a.Alloc(b.GetN(), b.GetOrder())
	if a.Order >= 1 {
		for i := 0; i < b.GetN(); i++ {
			a.Derivative[i] = b.GetDerivative(i)
		}
		if a.Order >= 2 {
			for i := 0; i < b.GetN(); i++ {
				for j := 0; j < b.GetN(); j++ {
					a.Hessian[i][j] = b.GetHessian(i, j)
				}
			}
		}
	}
}

// Set the value of the variable. All derivatives are reset to zero.
func (a *BasicState) SetValue(v float64) {
	a.Value = v
	a.ResetDerivatives()
}

func (a *BasicState) setValue(v float64) {
	a.Value = v
}

// Set the derivative of the ith variable to v.
func (a *BasicState) SetDerivative(i int, v float64) {
	a.Derivative[i] = v
}

func (a *BasicState) SetHessian(i, j int, v float64) {
	a.Hessian[i][j] = v
}

// Allocate memory for n variables and set the derivative
// of the ith variable to 1 (initial value).
func (a *BasicState) SetVariable(i, n, order int) error {
	if order > 2 {
		return fmt.Errorf("order `%d' not supported by this type", order)
	}
	a.Alloc(n, order)
	if order > 0 {
		a.Derivative[i] = 1
	}
	return nil
}

/* json
 * -------------------------------------------------------------------------- */

func (obj *BasicState) MarshalJSON() ([]byte, error) {
	t1 := false
	t2 := false
	if obj.Order > 0 && obj.N > 0 {
		// check for non-zero derivatives
		for i := 0; !t1 && i < obj.GetN(); i++ {
			if obj.Derivative[i] != 0.0 {
				t1 = true
			}
		}
		if obj.Order > 1 {
			// check for non-zero second derivatives
			for i := 0; !t2 && i < obj.GetN(); i++ {
				for j := 0; !t2 && j < obj.GetN(); j++ {
					if obj.GetHessian(i, j) != 0.0 {
						t2 = true
					}
				}
			}
		}
	}
	if t1 && t2 {
		r := struct {
			Value      float64
			Derivative []float64
			Hessian    [][]float64
		}{
			obj.Value, obj.Derivative, obj.Hessian}
		return json.Marshal(r)
	} else if t1 && !t2 {
		r := struct {
			Value      float64
			Derivative []float64
		}{
			obj.Value, obj.Derivative}
		return json.Marshal(r)
	} else if !t1 && t2 {
		r := struct {
			Value   float64
			Hessian [][]float64
		}{
			obj.Value, obj.Hessian}
		return json.Marshal(r)
	} else {
		return json.Marshal(obj.Value)
	}
}

func (obj *BasicState) UnmarshalJSON(data []byte) error {
	r := struct {
		Value      float64
		Derivative []float64
		Hessian    [][]float64
	}{}
	if err := json.Unmarshal(data, &r); err == nil {
		obj.Value = r.Value
		if len(r.Derivative) != 0 && len(r.Hessian) != 0 {
			if len(r.Derivative) != len(r.Derivative) {
				return fmt.Errorf("invalid json scalar representation")
			}
			obj.Alloc(len(r.Derivative), 2)
			obj.Derivative = r.Derivative
			obj.Hessian = r.Hessian
		} else if len(r.Derivative) != 0 && len(r.Hessian) == 0 {
			obj.Alloc(len(r.Derivative), 1)
			obj.Derivative = r.Derivative
		} else if len(r.Derivative) == 0 && len(r.Hessian) != 0 {
			obj.Alloc(len(r.Derivative), 2)
			obj.Hessian = r.Hessian
		}
		return nil
	} else {
		return json.Unmarshal(data, &obj.Value)
	}
}
