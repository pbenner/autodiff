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

package main

/* -------------------------------------------------------------------------- */

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type Line struct {
  slope     Scalar
  intercept Scalar
}

func NewLine(slope, intercept Scalar) *Line {

  l := new(Line)
  l.slope     = slope
  l.intercept = intercept

  return l
}

func (l *Line) Slope() Scalar {
  return l.slope
}

func (l *Line) Intercept() Scalar {
  return l.intercept
}

func (l *Line) SetSlope(s Scalar) {
  l.slope = s
}

func (l *Line) SetIntercept(i Scalar) {
  l.intercept = i
}

func (l *Line) Eval(x Scalar) Scalar {

  return Add(Mul(l.slope, x), l.intercept)
}
