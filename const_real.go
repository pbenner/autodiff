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
import "math"

/* -------------------------------------------------------------------------- */

type ConstReal float64

/* constructors
 * -------------------------------------------------------------------------- */

func NewConstReal(v float64) ConstReal {
  return ConstReal(v)
}

func NullConstReal() ConstReal {
  return ConstReal(0.0)
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (a ConstReal) String() string {
  return fmt.Sprintf("%e", a.GetValue())
}

/* read access
 * -------------------------------------------------------------------------- */

func (a ConstReal) GetOrder() int {
  return 0
}

func (a ConstReal) GetValue() float64 {
  return float64(a)
}

func (a ConstReal) GetLogValue() float64 {
  return math.Log(a.GetValue())
}

func (a ConstReal) GetDerivative(i int) float64 {
  return 0.0
}

func (a ConstReal) GetHessian(i, j int) float64 {
  return 0.0
}

func (a ConstReal) GetN() int {
  return 0
}
