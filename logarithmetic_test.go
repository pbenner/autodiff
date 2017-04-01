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

import "math"
import "testing"

/* -------------------------------------------------------------------------- */

func TestLogAdd1(t *testing.T) {

  a := math.Log(2.3)
  b := math.Log(1.2)
  c := logAdd(a,b)

  if math.Abs(math.Exp(c) - (2.3+1.2)) > 1e-8 {
    t.Error("logAdd() failed!")
  }
}

func TestLogAdd2(t *testing.T) {

  a := math.Log(0.0)
  b := math.Log(1.2)
  c := logAdd(a,b)

  if math.Abs(math.Exp(c) - (0.0+1.2)) > 1e-8 {
    t.Error("logAdd() failed!")
  }
}

func TestLogAdd3(t *testing.T) {

  a := math.Log(2.3)
  b := math.Log(0.0)
  c := logAdd(a,b)

  if math.Abs(math.Exp(c) - (2.3+0.0)) > 1e-8 {
    t.Error("logAdd() failed!")
  }
}

func TestLogSub1(t *testing.T) {

  a := math.Log(2.3)
  b := math.Log(1.2)
  c := logSub(a,b)

  if math.Abs(math.Exp(c) - (2.3-1.2)) > 1e-8 {
    t.Error("logAdd() failed!")
  }
}

func TestLogSub2(t *testing.T) {

  a := math.Log(2.3)
  b := math.Log(0.0)
  c := logSub(a,b)

  if math.Abs(math.Exp(c) - (2.3-0.0)) > 1e-8 {
    t.Error("logAdd() failed!")
  }
}
