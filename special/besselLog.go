/* Copyright (C) 2017 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/bessel.hpp
 */

//  Copyright (c) 2006 Xiaogang Zhang
//  Copyright (c) 2007 John Maddock
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

package special

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

func bessel_i0_log(x float64) float64 {
  if logx := math.Log(x); x < 7.75 {
    // Bessel I0 over[10 ^ -16, 7.75]
    // Max error in interpolated form : 3.042e-18
    // Max Error found at double precision = Poly : 5.106609e-16 Cheb : 5.239199e-16
    P := NewLogPolynomial([]float64{
      math.Log(1.00000000000000000e+00),
      math.Log(2.49999999999999909e-01),
      math.Log(2.77777777777782257e-02),
      math.Log(1.73611111111023792e-03),
      math.Log(6.94444444453352521e-05),
      math.Log(1.92901234513219920e-06),
      math.Log(3.93675991102510739e-08),
      math.Log(6.15118672704439289e-10),
      math.Log(7.59407002058973446e-12),
      math.Log(7.59389793369836367e-14),
      math.Log(6.27767773636292611e-16),
      math.Log(4.34709704153272287e-18),
      math.Log(2.63417742690109154e-20),
      math.Log(1.13943037744822825e-22),
      math.Log(9.07926920085624812e-25) })
    a := 2.0*logx - math.Log(4)
    return logAdd(a + P.Eval(a), 0.0)
  } else
  if x < 500 {
    // Max error in interpolated form : 1.685e-16
    // Max Error found at double precision = Poly : 2.575063e-16 Cheb : 2.247615e+00
    P := NewPolynomial([]float64{
       3.98942280401425088e-01,
       4.98677850604961985e-02,
       2.80506233928312623e-02,
       2.92211225166047873e-02,
       4.44207299493659561e-02,
       1.30970574605856719e-01,
      -3.35052280231727022e+00,
       2.33025711583514727e+02,
      -1.13366350697172355e+04,
       4.24057674317867331e+05,
      -1.23157028595698731e+07,
       2.80231938155267516e+08,
      -5.01883999713777929e+09,
       7.08029243015109113e+10,
      -7.84261082124811106e+11,
       6.76825737854096565e+12,
      -4.49034849696138065e+13,
       2.24155239966958995e+14,
      -8.13426467865659318e+14,
       2.02391097391687777e+15,
      -3.08675715295370878e+15,
      2.17587543863819074e+15 })
    return x + math.Log(P.Eval(1.0/x)) - 0.5*logx
  } else {
    // Max error in interpolated form : 2.437e-18
     // Max Error found at double precision = Poly : 1.216719e-16
    P := NewLogPolynomial([]float64{
      math.Log(3.98942280401432905e-01),
      math.Log(4.98677850491434560e-02),
      math.Log(2.80506308916506102e-02),
      math.Log(2.92179096853915176e-02),
      math.Log(4.53371208762579442e-02) })
    logex  := x / 2
    result := logex + P.Eval(-logx) - 0.5*logx
    result += logex
    return result
  }
}

/* -------------------------------------------------------------------------- */

func bessel_i1_log(x float64) float64 {
  if logx := math.Log(x); x < 7.75 {
    // Bessel I0 over[10 ^ -16, 7.75]
    // Max error in interpolated form: 5.639e-17
    // Max Error found at double precision = Poly: 1.795559e-16
    P := NewLogPolynomial([]float64{
      math.Log(8.333333333333333803e-02),
      math.Log(6.944444444444341983e-03),
      math.Log(3.472222222225921045e-04),
      math.Log(1.157407407354987232e-05),
      math.Log(2.755731926254790268e-07),
      math.Log(4.920949692800671435e-09),
      math.Log(6.834657311305621830e-11),
      math.Log(7.593969849687574339e-13),
      math.Log(6.904822652741917551e-15),
      math.Log(5.220157095351373194e-17),
      math.Log(3.410720494727771276e-19),
      math.Log(1.625212890947171108e-21),
      math.Log(1.332898928162290861e-23) })
    a := 2.0*logx - math.Log(4)
    Q := NewLogPolynomial([]float64{
      0.0, math.Log(0.5), P.Eval(a) })
    return logx + Q.Eval(a) - math.Log(2)
  } else
  if x < 500 {
      // Max error in interpolated form: 1.796e-16
      // Max Error found at double precision = Poly: 2.898731e-16
    P := NewPolynomial([]float64{
       3.989422804014406054e-01,
      -1.496033551613111533e-01,
      -4.675104253598537322e-02,
      -4.090895951581637791e-02,
      -5.719036414430205390e-02,
      -1.528189554374492735e-01,
       3.458284470977172076e+00,
      -2.426181371595021021e+02,
       1.178785865993440669e+04,
      -4.404655582443487334e+05,
       1.277677779341446497e+07,
      -2.903390398236656519e+08,
       5.192386898222206474e+09,
      -7.313784438967834057e+10,
       8.087824484994859552e+11,
      -6.967602516005787001e+12,
       4.614040809616582764e+13,
      -2.298849639457172489e+14,
       8.325554073334618015e+14,
      -2.067285045778906105e+15,
       3.146401654361325073e+15,
      -2.213318202179221945e+15 })
    return x + math.Log(P.Eval(1.0/x)) - 0.5*logx
   } else {
      // Max error in interpolated form: 1.320e-19
      // Max Error found at double precision = Poly: 7.065357e-17
     P := NewPolynomial([]float64{
        3.989422804014314820e-01,
       -1.496033551467584157e-01,
       -4.675105322571775911e-02,
       -4.090421597376992892e-02,
       -5.843630344778927582e-02 })
     ex     := x / 2
     result := ex + math.Log(P.Eval(1 / x)) - 0.5*logx
     result += ex
     return result
   }
}

/* -------------------------------------------------------------------------- */

type cyl_bessel_i_small_z_log struct {
  k    int
  v    float64
  term float64
  mult float64
}

func new_cyl_bessel_i_small_z_log(v, z float64) *cyl_bessel_i_small_z_log {
  r := cyl_bessel_i_small_z_log{}
  r.term = 0
  r.k    = 0
  r.v    = math.Log(v)
  r.mult = 2.0*math.Log(z) - math.Log(4)
  return &r
}

func (obj *cyl_bessel_i_small_z_log) Eval() float64 {
  r  := obj.term
  lk := math.Log(float64(obj.k+1))
  obj.k    += 1
  obj.term += obj.mult - lk
  obj.term -= logAdd(lk, obj.v)
  return r
}

/* -------------------------------------------------------------------------- */

func bessel_i_small_z_series_log(v, x float64) float64 {
  var prefix float64

  t, _  := math.Lgamma(v + 1)
  prefix = math.Log(x / 2)*v - t

  if math.IsInf(prefix, -1) {
    return prefix
  }

  s := new_cyl_bessel_i_small_z_log(v, x)

  return prefix + SumLogSeries(s, 0.0, math.Log(2.22045e-16), SeriesIterationsMax)
}

/* -------------------------------------------------------------------------- */

func bessel_i_log(v, x float64) float64 {
  //
  // This handles all the bessel I functions, note that we don't optimise
  // for integer v, other than the v = 0 or 1 special cases, as Millers
  // algorithm is at least as inefficient as the general case (the general
  // case has better error handling too).
  //
  if x < 0 {
    // better have integer v:
    if math.Floor(v) == v {
      if iround(v) & 1 != 0 {
        return math.NaN()
      } else {
        return bessel_i_log(v, -x)
      }
    } else {
      panic(fmt.Sprintf("Got x = %f, but we need x >= 0", x))
    }
  }
  if x == 0.0 {
    if v == 0.0 {
      return 0.0
    } else {
      return math.Inf(-1)
    }
  }
  if v == 0.5 {
    // common special case, note try and avoid overflow in exp(x):
    if x >= MaxLogFloat64 {
      e := x/2.0
      return e + (e - 0.5*math.Log(2.0*x*math.Pi))
    } else {
      return 0.5*(math.Log(2.0) - math.Log(x) - math.Log(math.Pi)) + math.Log(math.Sinh(x))
    }
  }
  if v == 0 {
    return bessel_i0_log(x)
  }
  if v == 1 {
    return bessel_i1_log(x)
  }
  if v > 0 && x / v < 0.25 {
    return bessel_i_small_z_series_log(v, x)
  }
  I, _ := bessel_ik(v, x, need_i)
  return math.Log(I)
}

/* -------------------------------------------------------------------------- */

// modified bessel function of the first kind
func LogBesselI(v, x float64) float64 {
  return bessel_i_log(v, x)
}
