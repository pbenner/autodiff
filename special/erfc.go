/* Copyright (C) 2016 Philipp Benner
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003 Gerard Jungman
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

/* Code ported from boost (http://www.gnu.org/software/gsl).
 * specfunc/erfc.c
 */

package special

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

/* See:
 * Hart et al., Computer Approximations (1968)
 */

func logErfc8(x float64) float64 {
  P := NewPolynomial([]float64{
    2.9788656263939928886200000000,
    7.4097406059647417944250000000,
    6.1602098531096305440906000000,
    5.0190497267842674634500580000,
    1.2753666447299659524795852640,
    0.5641895835477550741253201704 })
  Q := NewPolynomial([]float64{
    3.3690752069827527677000000000,
    9.6089653271927878706980000000,
    17.081440747466004315710950000,
    12.048951927855129036034049100,
    9.3960340162350541504305796480,
    2.2605285207673269695918669450,
    1.0000000000000000000000000000 })

  e := P.Eval(x)/Q.Eval(x)
  e  = math.Log(e) - x*x
  return e
}

func logErfc0(x float64) float64 {
  y := x / M_SQRTPI
  P := NewPolynomial([]float64{
     0.000000000000000000,
     1.000000000000000000,
     1.000000000000000000,
     (4.0 - math.Pi)/3.0,
     2.0*(1.0 - math.Pi/3.0),
    -0.001829764677455021,
     0.026296515210574650,
    -0.016215753788354040,
     0.001259939617621160,
     0.005569646491380000,
    -0.004556333980200000,
     0.000946158903200000,
     0.001320024317400000,
    -0.001429060000000000,
     0.000482040000000000 })
  return -2.0*P.Eval(y)
}

func LogErfc(x float64) float64 {

  if x*x < 2.4607833005759251e-02 {
    return logErfc0(x)
  } else if x > 8.0 {
    return logErfc8(x)
  } else {
    return math.Log(math.Erfc(x))
  }
}
