/* Copyright (C) 2017 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/bessel.hpp
 */

//  Copyright (c) 2006 Xiaogang Zhang
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

package special

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

const log_max_value float64 = 709.0

/* -------------------------------------------------------------------------- */

func iround(x float64) int {
  if x < 0 {
    return int(x-0.5)
  } else {
    return int(x+0.5)
  }
}

/* -------------------------------------------------------------------------- */

func modified_bessel_i0(x float64) float64 {
  P1 := NewPolynomial([]float64{
    -2.2335582639474375249e+15,
    -5.5050369673018427753e+14,
    -3.2940087627407749166e+13,
    -8.4925101247114157499e+11,
    -1.1912746104985237192e+10,
    -1.0313066708737980747e+08,
    -5.9545626019847898221e+05,
    -2.4125195876041896775e+03,
    -7.0935347449210549190e+00,
    -1.5453977791786851041e-02,
    -2.5172644670688975051e-05,
    -3.0517226450451067446e-08,
    -2.6843448573468483278e-11,
    -1.5982226675653184646e-14,
    -5.2487866627945699800e-18 })
  Q1 := NewPolynomial([]float64{
    -2.2335582639474375245e+15,
     7.8858692566751002988e+12,
    -1.2207067397808979846e+10,
     1.0377081058062166144e+07,
    -4.8527560179962773045e+03,
     1.00000000000000000000+00 })
  P2 := NewPolynomial([]float64{
    -2.2210262233306573296e-04,
     1.3067392038106924055e-02,
    -4.4700805721174453923e-01,
     5.5674518371240761397e+00,
    -2.3517945679239481621e+01,
     3.1611322818701131207e+01,
    -9.6090021968656180000e+00 })
  Q2 := NewPolynomial([]float64{
    -5.5194330231005480228e-04,
     3.2547697594819615062e-02,
    -1.1151759188741312645e+00,
     1.3982595353892851542e+01,
    -6.0228002066743340583e+01,
     8.5539563258012929600e+01,
    -3.1446690275135491500e+01,
     1.0000000000000000000e+00 })

  if x < 0 {
    // negative x is handled before we get here
    panic("internal error")
  }
  if x == 0 {
    return 1.0
  }
  if x <= 15 {
    // x in (0, 15]
    return P1.Eval(x*x) / Q1.Eval(x*x)
  } else {
    // x in (15, \infty)
    y := 1.0/x - 1.0/15.0
    r := P2.Eval(y) / Q2.Eval(y)
    return  math.Exp(x) / math.Sqrt(x) * r
  }
}

/* -------------------------------------------------------------------------- */

func modified_bessel_i1(x float64) float64 {
  P1 := NewPolynomial([]float64{
    -1.4577180278143463643e+15,
    -1.7732037840791591320e+14,
    -6.9876779648010090070e+12,
    -1.3357437682275493024e+11,
    -1.4828267606612366099e+09,
    -1.0588550724769347106e+07,
    -5.1894091982308017540e+04,
    -1.8225946631657315931e+02,
    -4.7207090827310162436e-01,
    -9.1746443287817501309e-04,
    -1.3466829827635152875e-06,
    -1.4831904935994647675e-09,
    -1.1928788903603238754e-12,
    -6.5245515583151902910e-16,
    -1.9705291802535139930e-19 })
  Q1 := NewPolynomial([]float64{
    -2.9154360556286927285e+15,
     9.7887501377547640438e+12,
    -1.4386907088588283434e+10,
     1.1594225856856884006e+07,
    -5.1326864679904189920e+03,
     1.0000000000000000000e+00 })
  P2 := NewPolynomial([]float64{
     1.4582087408985668208e-05,
    -8.9359825138577646443e-04,
     2.9204895411257790122e-02,
    -3.4198728018058047439e-01,
     1.3960118277609544334e+00,
    -1.9746376087200685843e+00,
     8.5591872901933459000e-01,
    -6.0437159056137599999e-02 })
  Q2 := NewPolynomial([]float64{
     3.7510433111922824643e-05,
    -2.2835624489492512649e-03,
     7.4212010813186530069e-02,
    -8.5017476463217924408e-01,
     3.2593714889036996297e+00,
    -3.8806586721556593450e+00,
     1.0000000000000000000e+00 })

  if x < 0 {
    // negative x is handled before we get here
    panic("internal error")
  }
  w := math.Abs(x)
  if x == 0.0 {
    return 0.0
  }
  if w <= 15 {
    // w in (0, 15]
    return w * P1.Eval(x*x) / Q1.Eval(x*x)
  } else {
    // w in (15, \infty)
    y := 1.0/w - 1.0/15
    return math.Exp(w) / math.Sqrt(w) * P2.Eval(y) / Q2.Eval(y)
  }
}

/* -------------------------------------------------------------------------- */

// Compute I(v, x) and K(v, x) simultaneously by Temme's method, see
// Temme, Journal of Computational Physics, vol 19, 324 (1975)
func modified_bessel_ik(v, x float64, kind int) (float64, float64) {
  // Kv1 = K_(v+1), fv = I_(v+1) / I_v
  // Ku1 = K_(u+1), fu = I_(u+1) / I_u
  var u, I, K, Iv, Kv, Kv1, Ku, Ku1, fv float64
  var W, current, prev, next float64
  var n, k int

  reflect  := false
  org_kind := kind

  if v < 0 {
    reflect = true
    v       = -v                            // v is non-negative from here
    kind   |= need_k
  }
  n = iround(v)
  u = v - float64(n)                        // -1/2 <= u < 1/2

  if x < 0 {
    panic(fmt.Sprintf("Got x = %f but real argument x must be non-negative, complex number result not supported.", x))
  }
  if x == 0.0 {
    if v == 0.0 {
      I = 1.0
    } else {
      I = 0.0
    }
    if kind & need_k {
      K = math.Inf(1)
    } else {
      K = math.NaN() // any value will do
    }
    if reflect && (kind & need_i) {
      z := u + float64(n % 2)
      if math.Sin(math.Pi*z) != 0.0 {
        I = math.Inf(1)
      }
    }
    return I, K
  }
  // x is positive until reflection
  W = 1.0/x                                  // Wronskian
  if x <= 2 {                                // x in (0, 2]
    Ku, Ku1 = temme_ik(u, x)                 // Temme series
  } else {                                   // x in (2, \infty)
    Ku, Ku1 = CF2_ik(u, x)                   // continued fraction CF2_ik
  }
  prev        = Ku
  current     = Ku1
  scale      := 1.0
  scale_sign := 1.0
  for k = 1; k <= n; k++ {                   // forward recurrence for K
    fact := 2.0*(u + float64(k)) / x
    if (math.MaxFloat64 - math.Abs(prev)) / fact < math.Abs(current) {
      prev  /= current
      scale /= current
      if current < 0 {
        scale_sign *= -1.0
      }
      current = 1
    }
    next    = fact * current + prev
    prev    = current
    current = next
  }
  Kv  = prev
  Kv1 = current

  if kind & need_i {
    lim := (4.0 * v * v + 10.0) / (8.0 * x)
    lim *= lim
    lim *= lim
    lim /= 24.0
    if (lim < EpsilonFloat64 * 10.0) && (x > 100.0) {
      // x is huge compared to v, CF1 may be very slow
      // to converge so use asymptotic expansion for large
      // x case instead.  Note that the asymptotic expansion
      // isn't very accurate - so it's deliberately very hard
      // to get here - probably we're going to overflow:
      Iv = asymptotic_bessel_i_large_x(v, x)
    } else
    if (v > 0.0) && (x / v < 0.25) {
      Iv = bessel_i_small_z_series(v, x)
    } else {
      fv = CF1_ik(v, x)                      // continued fraction CF1_ik
      Iv = scale * W / (Kv * fv + Kv1)       // Wronskian relation
    }
  } else {
    Iv = math.NaN() // any value will do
  }
  if reflect {
    z     = u + float64(n % 2)
    fact := 2.0 / math.Pi * math.Sin(math.Pi*z) * Kv
    if(fact == 0) {
      I = Iv
    } else
    if math.MaxFloat64 * scale < fact {
      if org_kind & need_i {
        if fact*scale_sign < 0 {
          I = math.Inf(-1)
        } else {
          I = math.Inf( 1)
        }
      } else {
        I = 0.0
      }
    } else {
      I = Iv + fact / scale   // reflection formula
    }
  } else {
    I = Iv
  }
  if math.MaxFloat64 * scale < Kv {
    if org_kind & need_k {
      if Kv * scale_sign < 0.0 {
        K = math.Inf(-1)
      } else {
        K = math.Inf( 1)
      }
    } else {
      K = 0.0
    }
  } else {
    K = Kv / scale
  }
  return I, K
}

/* -------------------------------------------------------------------------- */

func modified_bessel_i_imp(v, x float64) float64 {
  //
  // This handles all the bessel I functions, note that we don't optimise
  // for integer v, other than the v = 0 or 1 special cases, as Millers
  // algorithm is at least as inefficient as the general case (the general
  // case has better error handling too).
  //
  if x < 0 {
    // better have integer v:
    if math.Floor(v) == v {
      r := modified_bessel_i_imp(v, -x)
      if iround(v) & 1 != 0 {
        return -r
      } else {
        return  r
      }
    } else {
      panic(fmt.Sprintf("Got x = %f, but we need x >= 0", x))
    }
  }
  if x == 0.0 {
    if v == 0.0 {
      return 1.0
    } else {
      return 0.0
    }
  }
  if v == 0.5 {
    // common special case, note try and avoid overflow in exp(x):
    if x >= log_max_value {
      e := math.Exp(x/2.0)
      return e*(e/math.Sqrt(2.0*x*math.Pi))
    } else {
      return math.Sqrt(2.0/(x*math.Pi)) * math.Sinh(x)
    }
  }
  if v == 0 {
    return modified_bessel_i0(x)
  }
  if v == 1 {
    return modified_bessel_i1(x)
  }
  if v > 0 && x / v < 0.25 {
    return modified_bessel_i_small_z_series(v, x)
  }
  I, _ := modified_bessel_ik(v, x, need_i)
  return I
}
