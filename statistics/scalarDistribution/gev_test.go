/* Copyright (C) 2016-2020 Philipp Benner
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

package scalarDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestGevDistribution1(t *testing.T) {

  mu    := NewFloat64(1.0)
  sigma := NewFloat64(1.0)
  xi    := NewFloat64(0.0)

  gev, _ := NewGevDistribution(mu, sigma, xi)

  x := NewFloat64(100)
  y := NewFloat64(0.0)
  z := NewFloat64(0.0)

  gev.LogPdf(y, x)
  gev.Cdf(z, x)

  if math.Abs(y.GetFloat64() - -99.0) > 1e-4 {
    t.Error("Gev LogPdf failed!")
  }
  if math.Abs(z.GetFloat64() - 1.0) > 1e-4 {
    t.Error("Gev LogPdf failed!")
  }
}

func TestGevDistribution2(t *testing.T) {

  mu    := NewFloat64(1.0)
  sigma := NewFloat64(1.0)
  xi    := NewFloat64(0.0)

  gev, _ := NewGevDistribution(mu, sigma, xi)

  x := NewFloat64(10)
  y := NewFloat64(0.0)
  z := NewFloat64(0.0)

  gev.LogPdf(y, x)
  gev.Cdf(z, x)

  if math.Abs(y.GetFloat64() - -9.000123) > 1e-4 {
    t.Error("Gev LogPdf failed!")
  }
  if math.Abs(z.GetFloat64() - 0.9998766) > 1e-4 {
    t.Error("Gev LogPdf failed!")
  }
}

func TestGevDistribution3(t *testing.T) {

  mu    := NewFloat64(1.0)
  sigma := NewFloat64(1.0)
  xi    := NewFloat64(2.0)

  gev, _ := NewGevDistribution(mu, sigma, xi)

  x := NewFloat64(100)
  y := NewFloat64(0.0)
  z := NewFloat64(0.0)

  gev.LogPdf(y, x)
  gev.Cdf(z, x)

  if math.Abs(y.GetFloat64() - -8.010845) > 1e-4 {
    t.Error("Gev LogPdf failed!")
  }
  if math.Abs(z.GetFloat64() - 0.9315661) > 1e-4 {
    t.Error("Gev LogPdf failed!")
  }
}
