/* Copyright (C) 2016 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/zeta.hpp
 */

//  Copyright John Maddock 2007, 2014.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

package special

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

func init() {
  zeta_imp_odd_integer_init()
}

/* -------------------------------------------------------------------------- */

func zeta_polynomial_series(s, sc float64) float64 {
  //
  // This is algorithm 3 from:
  // 
  // "An Efficient Algorithm for the Riemann Zeta Function", P. Borwein, 
  // Canadian Mathematical Society, Conference Proceedings.
  // See: http://www.cecm.sfu.ca/personal/pborwein/PAPERS/P155.pdf
  //
  n       := int(math.Trunc(math.Log(EpsilonFloat64) / -2.0))
  sum     := 0.0
  two_n   := math.Pow(2.0, float64(n))
  ej_sign := 1.0

  for j := 0; j < n; j++ {
    sum    +=  ej_sign * -two_n / math.Pow(float64(j + 1), s)
    ej_sign = -ej_sign
  }
  ej_sum  := 1.0
  ej_term := 1.0
  for j := n; j <= 2 * n - 1; j++ {
    sum     +=  ej_sign * (ej_sum - two_n) / math.Pow(float64(j + 1), s)
    ej_sign  = -ej_sign
    ej_term *= float64(2 * n - j)
    ej_term /= float64(j - n + 1)
    ej_sum  += ej_term
  }
  return -sum / (two_n * (-Powm1(2.0, sc)))
}

func zeta_imp_prec(s, sc float64) float64 {
  var result float64

  if s < 1.0 {
    // Rational Approximation
    // Maximum Deviation Found:                     2.020e-18
    // Expected Error Term:                        -2.020e-18
    // Max error found at double precision:         3.994987e-17
    P := NewPolynomial([]float64{
       0.24339294433593750202,
      -0.49092470516353571651,
       0.0557616214776046784287,
      -0.00320912498879085894856,
       0.000451534528645796438704,
      -0.933241270357061460782e-5,
    })
    Q := NewPolynomial([]float64{
       1.0,
      -0.279960334310344432495,
       0.0419676223309986037706,
      -0.00413421406552171059003,
       0.00024978985622317935355,
      -0.101855788418564031874e-4 })
    result  = P.Eval(sc) / Q.Eval(sc)
    result -= 1.2433929443359375
    result += sc
    result /= sc
  } else if(s <= 2.0) {
    // Maximum Deviation Found:        9.007e-20
    // Expected Error Term:            9.007e-20
    P := NewPolynomial([]float64{
       0.577215664901532860516,
       0.243210646940107164097,
       0.0417364673988216497593,
       0.00390252087072843288378,
       0.000249606367151877175456,
       0.110108440976732897969e-4 })
    Q := NewPolynomial([]float64{
       1.0,
       0.295201277126631761737,
       0.043460910607305495864,
       0.00434930582085826330659,
       0.000255784226140488490982,
       0.10991819782396112081e-4 })
    result  = P.Eval(-sc) / Q.Eval(-sc)
    result += 1 / (-sc);
  } else if(s <= 4.0) {
    // Maximum Deviation Found:          5.946e-22
    // Expected Error Term:             -5.946e-22
    const Y = 0.6986598968505859375
    P := NewPolynomial([]float64{
      -0.0537258300023595030676,
       0.0445163473292365591906,
       0.0128677673534519952905,
       0.00097541770457391752726,
       0.769875101573654070925e-4,
       0.328032510000383084155e-5 })
    Q := NewPolynomial([]float64{
       1.0,
       0.33383194553034051422,
       0.0487798431291407621462,
       0.00479039708573558490716,
       0.000270776703956336357707,
       0.106951867532057341359e-4,
       0.236276623974978646399e-7 })
    result  = P.Eval(s - 2.0) / Q.Eval(s - 2.0)
    result += Y + 1.0 / (-sc)
  } else if(s <= 7.0) {
    // Maximum Deviation Found:                     2.955e-17
    // Expected Error Term:                         2.955e-17
    // Max error found at double precision:         2.009135e-16
    P := NewPolynomial([]float64{
      -2.49710190602259410021,
      -2.60013301809475665334,
      -0.939260435377109939261,
      -0.138448617995741530935,
      -0.00701721240549802377623,
      -0.229257310594893932383e-4 })
    Q := NewPolynomial([]float64{
       1.0,
       0.706039025937745133628,
       0.15739599649558626358,
       0.0106117950976845084417,
      -0.36910273311764618902e-4,
       0.493409563927590008943e-5,
      -0.234055487025287216506e-6,
       0.718833729365459760664e-8,
      -0.1129200113474947419e-9 })
      result = P.Eval(s - 4.0) / Q.Eval(s - 4.0)
      result = 1.0 + math.Exp(result)
   } else if(s < 15.0) {
     // Maximum Deviation Found:                     7.117e-16
     // Expected Error Term:                         7.117e-16
     // Max error found at double precision:         9.387771e-16
     P := NewPolynomial([]float64{
       -4.78558028495135619286,
       -1.89197364881972536382,
       -0.211407134874412820099,
       -0.000189204758260076688518,
        0.00115140923889178742086,
        0.639949204213164496988e-4,
        0.139348932445324888343e-5 })
     Q := NewPolynomial([]float64{
       1.0,
        0.244345337378188557777,
        0.00873370754492288653669,
       -0.00117592765334434471562,
       -0.743743682899933180415e-4,
       -0.21750464515767984778e-5,
        0.471001264003076486547e-8,
       -0.833378440625385520576e-10,
        0.699841545204845636531e-12 })
     result = P.Eval(s - 7.0) / Q.Eval(s - 7.0);
     result = 1 + math.Exp(result);
   } else if(s < 36.0) {
     // Max error in interpolated form:             1.668e-17
     // Max error found at long double precision:   1.669714e-17
     P := NewPolynomial([]float64{
       -10.3948950573308896825,
       -2.85827219671106697179,
       -0.347728266539245787271,
       -0.0251156064655346341766,
       -0.00119459173416968685689,
       -0.382529323507967522614e-4,
       -0.785523633796723466968e-6,
       -0.821465709095465524192e-8 })
     Q := NewPolynomial([]float64{
        1.0,
        0.208196333572671890965,
        0.0195687657317205033485,
        0.00111079638102485921877,
        0.408507746266039256231e-4,
        0.955561123065693483991e-6,
        0.118507153474022900583e-7,
        0.222609483627352615142e-14 })
     result = P.Eval(s - 15.0) / Q.Eval(s - 15.0);
     result = 1.0 + math.Exp(result);
   } else if s < 56.0 {
     result = 1.0 + math.Pow(2.0, -s);
   } else {
     result = 1.0
   }
   return result
}

// precompute zeta_imp_odd_integer and store the results
// in this array
var zeta_imp_odd_integer_results [50]float64

func zeta_imp_odd_integer_init() {
  results := &zeta_imp_odd_integer_results
  for k := 0; k < len(results); k++ {
      arg := k * 2 + 3;
    c_arg := 1 - arg;
    results[k] = zeta_polynomial_series(float64(arg), float64(c_arg))
  }
}

func zeta_imp_odd_integer(s int, sc float64) float64 {
  results := &zeta_imp_odd_integer_results
  index   := (s - 3) / 2
  if index >= len(results) {
    return zeta_polynomial_series(float64(s), sc)
  } else {
    return results[index]
  }
}

func zeta_imp(s, sc float64) float64 {
  const rootEpsilon = 1.49012e-08
  const log_root_two_pi = 9.189385332046727417803297364056176398e-01

  if sc == 0 {
    // lim_{x->1} Zeta[x] = +/-Inf
    return math.NaN()
  }
  result := 0.0
  //
  // Trivial case:
  //
  if s > float64(PrecisionFloat64) {
    return 1.0
  }
  //
  // Start by seeing if we have a simple closed form:
  //
  if math.Floor(s) == s {
    v := int(math.Trunc(s))
    if float64(v) == s {
      if v < 0 {
        if (-v & 1) == 1 {
          return -BernoulliNumber(1-v)/float64(1-v)
        } else {
          return 0.0
        }
      } else if (v & 1) == 0 {
        if ((v / 2 - 1) & 1) == 1 {
          return -math.Pow(2.0, float64(v - 1))*math.Pow(math.Pi, float64(v))*BernoulliNumber(v)/Factorial(v)
        } else {
          return  math.Pow(2.0, float64(v - 1))*math.Pow(math.Pi, float64(v))*BernoulliNumber(v)/Factorial(v)
        }
      } else {
        return zeta_imp_odd_integer(v, sc)
      }
    }
  }

  if math.Abs(s) < rootEpsilon {
    result = -0.5 - log_root_two_pi * s;
  } else if(s < 0) {
    s, sc = sc, s
    if math.Floor(sc/2.0) == sc/2.0 {
      result = 0.0
    } else {
      if s > float64(factorialMax) {
        mult   := math.Sin(math.Pi*0.5*sc)*2.0*zeta_imp(s, sc)
        v, _   := math.Lgamma(s)
        result  = v
        result -= s * math.Log(2.0*math.Pi)
        if result > MaxLogFloat64 {
          if math.Signbit(mult) {
            return math.Inf(-1)
          } else {
            return math.Inf(+1)
          }
        }
        result = math.Exp(result)
        if math.MaxFloat64 / math.Abs(mult) < result {
          if math.Signbit(mult) {
            return math.Inf(-1)
          } else {
            return math.Inf(+1)
          }
        }
        result *= mult
      } else {
        result = math.Sin(math.Pi*0.5*sc) * 2 * math.Pow(2.0*math.Pi, -s) * math.Gamma(s) * zeta_imp(s, sc)
      }
    }
  } else {
    result = zeta_imp_prec(s, sc)
  }
  return result
}

/* -------------------------------------------------------------------------- */

func Zeta(s float64) float64 {
  return zeta_imp(s, 1.0-s)
}
