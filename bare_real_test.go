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
import "testing"

/* -------------------------------------------------------------------------- */

func TestBareDiff1(t *testing.T) {

  defer func() {
    if r := recover(); r == nil {
      t.Errorf("TestBareDiff1() failed!")
    }
  }()

  x := NewReal(9)
  y := NewBareReal(3)

  Variables(1, x)

  // this should panic
  Mul(y, x)
}

func TestBareDiff2(t *testing.T) {

  defer func() {
    if r := recover(); r != nil {
      t.Errorf("TestBareDiff2() failed!")
    }
  }()

  x := NewReal(9)
  y := NewBareReal(3)

  Variables(1, x)

  // this should be ok
  Mul(x, y)
}
