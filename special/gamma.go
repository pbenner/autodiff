/* Copyright (C) 2016 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/gamma.hpp
 */

//  Copyright John Maddock 2006-7, 2013-14.
//  Copyright Paul A. Bristow 2007, 2013-14.
//  Copyright Nikhar Agrawal 2013-14
//  Copyright Christopher Kormanyos 2013-14

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

package special

/* -------------------------------------------------------------------------- */

import "math"

/* -------------------------------------------------------------------------- */

type SmallGamma2Series struct {
  result float64
  x      float64
  apn    float64
  n      int
}

func NewSmallGamma2Series(a, x float64) *SmallGamma2Series {
  return &SmallGamma2Series{-x, -x, a+1.0, 1}
}

func (series *SmallGamma2Series) Eval() float64 {
  // result at this step
  r := series.result/series.apn
  // update variables
  series.result *= series.x
  series.n      += 1
  series.result /= float64(series.n)
  series.apn    += 1.0
  return r
}

/* -------------------------------------------------------------------------- */

type LowerIncompleteGammaSeries struct {
  result float64
  a      float64
  z      float64
}

func NewLowerIncompleteGammaSeries(a1, z1 float64) *LowerIncompleteGammaSeries {
  return &LowerIncompleteGammaSeries{1, a1, z1}
}

func (series *LowerIncompleteGammaSeries) Eval() float64 {
  r := series.result
  series.a      += 1.0
  series.result *= series.z/series.a
  return r
}

/* -------------------------------------------------------------------------- */

type UpperIncompleteGammaFraction struct {
  a float64
  z float64
  k int
}

func NewUpperIncompleteGammaFraction(a1, z1 float64) *UpperIncompleteGammaFraction {
  return &UpperIncompleteGammaFraction{a1, z1-a1+1.0, 0}
}

func (fraction *UpperIncompleteGammaFraction) Eval() (float64, float64) {
  fraction.k += 1
  fraction.z += 2.0
  return float64(fraction.k)*(fraction.a - float64(fraction.k)), fraction.z
}

/* -------------------------------------------------------------------------- */

func finite_gamma_q(a, x float64) float64 {
  //
  // Calculates normalised Q when a is an integer:
  //
  e := math.Exp(-x)
  sum := e
  if(sum != 0) {
    term := sum
    for n := 1.0; n < a; n += 1.0 {
      term /= n
      term *= x
      sum  += term
    }
  }
  return sum
}

func finite_half_gamma_q(a, x float64) float64 {
  //
  // Calculates normalised Q when a is a half-integer:
  //
  e := math.Erfc(math.Sqrt(x))
  if e != 0.0 && a > 1.0 {
    term := math.Exp(-x) / math.Sqrt(math.Pi*x)
    term *= x
    half := 1.0/2.0
    term /= half
    sum  := term
    for n := 2.0; n < a; n += 1.0 {
      term /= n - half
      term *= x
      sum  += term
    }
    e += sum
  }
  return e
}

func lower_gamma_series(a, z, init_value float64) float64 {
  s := NewLowerIncompleteGammaSeries(a, z)
  return SumSeries(s, init_value, 2.22045e-16, SeriesIterationsMax)
}

func upper_gamma_fraction(a, z float64) float64 {
  f := NewUpperIncompleteGammaFraction(a, z)
  return 1.0/(z - a + 1.0 + EvalContinuedFraction(f, 2.22045e-16, SeriesIterationsMax))
}

func regularised_gamma_prefix(a, z float64) float64 {
  limit := math.Max(10.0, a)
  sum   := lower_gamma_series  (a, limit, 0.0)/a
  sum   += upper_gamma_fraction(a, limit)

  if a < 10.0 {
    // special case for small a:
    prefix := math.Pow(z/10.0, a)
    prefix *= math.Exp(10.0 - z)
    if 0.0 == prefix {
      prefix = math.Pow((z*math.Exp((10.0-z)/a))/10.0, a)
    }
    prefix /= sum
    return prefix
  }

  zoa    := z/a
  amz    := a - z
  alzoa  := a*math.Log(zoa)
  prefix := 0.0
  if math.Min(alzoa, amz) <= MinLogFloat64 || math.Max(alzoa, amz) >= MaxLogFloat64 {
    amza := amz / a
    if amza <= MinLogFloat64 || amza >= MaxLogFloat64 {
      prefix = math.Exp(alzoa + amz)
    } else {
      prefix = math.Pow(zoa*math.Exp(amza), a)
    }
  } else {
    prefix = math.Pow(zoa, a)*math.Exp(amz)
  }
  prefix /= sum
  return prefix
}

func full_igamma_prefix(a, z float64) float64 {
  prefix := 0.0
  alz    := a*math.Log(z)

  if z >= 1 {
    if alz < MaxLogFloat64 && -z > MinLogFloat64 {
      prefix = math.Pow(z, a)*math.Exp(-z)
    } else
    if a >= 1.0 {
      prefix = math.Pow(z/math.Exp(z/a), a)
    } else {
      prefix = math.Exp(alz - z)
    }
  } else {
    if alz > MinLogFloat64 {
      prefix = math.Pow(z, a)*math.Exp(-z)
    } else
    if z/a < MaxLogFloat64 {
      prefix = math.Pow(z/math.Exp(z/a), a)
    } else {
      prefix = math.Exp(alz - z)
    }
   }

   return prefix
}

func lgamma_small_imp(z, zm1, zm2 float64) float64 {
  // This version uses rational approximations for small
  // values of z accurate enough for 64-bit mantissas
  // (80-bit long doubles), works well for 53-bit doubles as well.

  result := 0.0
  if z < EpsilonFloat64 {
    result = -math.Log(z)
  } else
  if zm1 == 0.0 || zm2 == 0.0 {
      // nothing to do, result is zero....
  } else
  if z > 2.0 {
    //
    // Begin by performing argument reduction until
    // z is in [2,3):
    //
    if z >= 3.0 {
      for {
        z      -= 1.0
        zm2    -= 1.0
        result += math.Log(z)
        if z >= 3.0 {
          break
        }
      }
      // Update zm2, we need it below:
      zm2 = z - 2
    }

    //
    // Use the following form:
    //
    // lgamma(z) = (z-2)(z+1)(Y + R(z-2))
    //
    // where R(z-2) is a rational approximation optimised for
    // low absolute error - as long as it's absolute error
    // is small compared to the constant Y - then any rounding
    // error in it's computation will get wiped out.
    //
    // R(z-2) has the following properties:
    //
    // At double: Max error found:                    4.231e-18
    // At long double: Max error found:               1.987e-21
    // Maximum Deviation Found (approximation error): 5.900e-24
    //
    P := NewPolynomial([]float64{
      -0.180355685678449379109e-1,
       0.25126649619989678683e-1,
       0.494103151567532234274e-1,
       0.172491608709613993966e-1,
      -0.259453563205438108893e-3,
      -0.541009869215204396339e-3,
      -0.324588649825948492091e-4 })
    Q := NewPolynomial([]float64{
       0.1e1,
       0.196202987197795200688e1,
       0.148019669424231326694e1,
       0.541391432071720958364e0,
       0.988504251128010129477e-1,
       0.82130967464889339326e-2,
       0.224936291922115757597e-3,
      -0.223352763208617092964e-6 })

    Y := 0.158963680267333984375

    r := zm2 * (z + 1)
    R := P.Eval(zm2)
    R /= Q.Eval(zm2)

    result +=  r*Y + r*R
  } else {
    //
    // If z is less than 1 use recurrance to shift to
    // z in the interval [1,2]:
    //
    if z < 1.0 {
      result += -math.Log(z)
      zm2 = zm1
      zm1 = z
      z  += 1.0
    }
    //
    // Two approximations, on for z in [1,1.5] and
    // one for z in [1.5,2]:
    //
    if z <= 1.5 {
      //
      // Use the following form:
      //
      // lgamma(z) = (z-1)(z-2)(Y + R(z-1))
      //
      // where R(z-1) is a rational approximation optimised for
      // low absolute error - as long as it's absolute error
      // is small compared to the constant Y - then any rounding
      // error in it's computation will get wiped out.
      //
      // R(z-1) has the following properties:
      //
      // At double precision: Max error found:                1.230011e-17
      // At 80-bit long double precision:   Max error found:  5.631355e-21
      // Maximum Deviation Found:                             3.139e-021
      // Expected Error Term:                                 3.139e-021

      //
      Y := 0.52815341949462890625

      P := NewPolynomial([]float64{
         0.490622454069039543534e-1,
        -0.969117530159521214579e-1,
        -0.414983358359495381969e0,
        -0.406567124211938417342e0,
        -0.158413586390692192217e0,
        -0.240149820648571559892e-1,
        -0.100346687696279557415e-2 })
      Q := NewPolynomial([]float64{
         0.1e1,
         0.302349829846463038743e1,
         0.348739585360723852576e1,
         0.191415588274426679201e1,
         0.507137738614363510846e0,
         0.577039722690451849648e-1,
         0.195768102601107189171e-2 })

      r := P.Eval(zm1)/Q.Eval(zm1)
      prefix := zm1*zm2

      result += prefix*Y + prefix*r
    } else {
      //
      // Use the following form:
      //
      // lgamma(z) = (2-z)(1-z)(Y + R(2-z))
      //
      // where R(2-z) is a rational approximation optimised for
      // low absolute error - as long as it's absolute error
      // is small compared to the constant Y - then any rounding
      // error in it's computation will get wiped out.
      //
      // R(2-z) has the following properties:
      //
      // At double precision, max error found:              1.797565e-17
      // At 80-bit long double precision, max error found:  9.306419e-21
      // Maximum Deviation Found:                           2.151e-021
      // Expected Error Term:                               2.150e-021
      //
      Y := 0.452017307281494140625

      P := NewPolynomial([]float64{
        -0.292329721830270012337e-1, 
         0.144216267757192309184e0,
        -0.142440390738631274135e0,
         0.542809694055053558157e-1,
        -0.850535976868336437746e-2,
         0.431171342679297331241e-3 })
      Q := NewPolynomial([]float64{
         0.1e1,
        -0.150169356054485044494e1,
         0.846973248876495016101e0,
        -0.220095151814995745555e0,
         0.25582797155975869989e-1,
        -0.100666795539143372762e-2,
        -0.827193521891290553639e-6 })
      r := zm2 * zm1
      R := P.Eval(-zm2)/Q.Eval(-zm2)

      result += r*Y + r*R
      }
   }
   return result
}

func igamma_temme_large(a, x float64) float64 {
  sigma := (x - a)/a
  phi   := -math.Log1p(sigma) - sigma
  y     := a*phi
  z     := math.Sqrt(2.0*phi)
  if x < a {
    z = -z
  }
  workspace := make([]float64, 10)

  C0 := NewPolynomial([]float64{
    -0.33333333333333333,
     0.083333333333333333,
    -0.014814814814814815,
     0.0011574074074074074,
     0.0003527336860670194,
    -0.00017875514403292181,
     0.39192631785224378e-4,
    -0.21854485106799922e-5,
    -0.185406221071516e-5,
     0.8296711340953086e-6,
    -0.17665952736826079e-6,
     0.67078535434014986e-8,
     0.10261809784240308e-7,
    -0.43820360184533532e-8,
     0.91476995822367902e-9 })
  workspace[0] = C0.Eval(z)

  C1 := NewPolynomial([]float64{
    -0.0018518518518518519,
    -0.0034722222222222222,
     0.0026455026455026455,
    -0.00099022633744855967,
     0.00020576131687242798,
    -0.40187757201646091e-6,
    -0.18098550334489978e-4,
     0.76491609160811101e-5,
    -0.16120900894563446e-5,
     0.46471278028074343e-8,
     0.1378633446915721e-6,
    -0.5752545603517705e-7,
     0.11951628599778147e-7 })
  workspace[1] = C1.Eval(z)

  C2 := NewPolynomial([]float64{
     0.0041335978835978836,
    -0.0026813271604938272,
     0.00077160493827160494,
     0.20093878600823045e-5,
    -0.00010736653226365161,
     0.52923448829120125e-4,
    -0.12760635188618728e-4,
     0.34235787340961381e-7,
     0.13721957309062933e-5,
    -0.6298992138380055e-6,
     0.14280614206064242e-6 })
  workspace[2] = C2.Eval(z)

  C3 := NewPolynomial([]float64{
     0.00064943415637860082,
     0.00022947209362139918,
    -0.00046918949439525571,
     0.00026772063206283885,
    -0.75618016718839764e-4,
    -0.23965051138672967e-6,
     0.11082654115347302e-4,
    -0.56749528269915966e-5,
     0.14230900732435884e-5 })
  workspace[3] = C3.Eval(z)

  C4 := NewPolynomial([]float64{
    -0.0008618882909167117,
     0.00078403922172006663,
    -0.00029907248030319018,
    -0.14638452578843418e-5,
     0.66414982154651222e-4,
    -0.39683650471794347e-4,
     0.11375726970678419e-4 })
   workspace[4] = C4.Eval(z)

  C5 := NewPolynomial([]float64{
    -0.00033679855336635815,
    -0.69728137583658578e-4,
     0.00027727532449593921,
    -0.00019932570516188848,
     0.67977804779372078e-4,
     0.1419062920643967e-6,
    -0.13594048189768693e-4,
     0.80184702563342015e-5,
    -0.22914811765080952e-5 })
  workspace[5] = C5.Eval(z)

  C6 := NewPolynomial([]float64{
     0.00053130793646399222,
    -0.00059216643735369388,
     0.00027087820967180448,
     0.79023532326603279e-6,
    -0.81539693675619688e-4,
     0.56116827531062497e-4,
    -0.18329116582843376e-4 })
  workspace[6] = C6.Eval(z)

  C7 := NewPolynomial([]float64{
     0.00034436760689237767,
     0.51717909082605922e-4,
    -0.00033493161081142236,
     0.0002812695154763237,
    -0.00010976582244684731 })
  workspace[7] = C7.Eval(z)

  C8 := NewPolynomial([]float64{
    -0.00065262391859530942,
     0.00083949872067208728,
    -0.00043829709854172101 })
  workspace[8] = C8.Eval(z)
  workspace[9] = -0.00059676129019274625

  result := NewPolynomial(workspace).Eval(1.0/a)
  result *= math.Exp(-y)/math.Sqrt(2.0*math.Pi*a)
  if x < a {
    result = -result
  }
  result += math.Erfc(math.Sqrt(y))/2.0

  return result
}

func tgammap1m1_imp(dz float64) float64 {

  result := 0.0

  if dz < 0.0 {
    if dz < -0.5 {
      // Best method is simply to subtract 1 from tgamma:
      result = math.Gamma(1+dz) - 1.0
    } else {
      // Use expm1 on lgamma:
      result = math.Expm1(-math.Log1p(dz) + lgamma_small_imp(dz+2.0, dz+1.0, dz))
    }
  } else {
    if dz < 2 {
      // Use expm1 on lgamma:
      result = math.Expm1(lgamma_small_imp(dz+1.0, dz, dz-1.0))
    } else {
      // Best method is simply to subtract 1 from tgamma:
      result = math.Gamma(dz+1.0) - 1.0
    }
  }
  return result
}

func tgamma1pm1(z float64) float64 {
  return tgammap1m1_imp(z)
}

func tgamma_small_upper_part(a, x float64, invert bool) (float64, float64) {
  //
  // Compute the full upper fraction (Q) when a is very small:
  //
  result := tgamma1pm1(a)
  pgam   := (result + 1.0)/a
  p      := Powm1(x, a)
  result -= p
  result /= a
  s := NewSmallGamma2Series(a, x)
  p += 1.0
  init_value := 0.0
  if invert {
    init_value = pgam
  }
  result = -p * SumSeries(s, (init_value - result)/p, 2.22045e-16, SeriesIterationsMax)
  if invert {
    result = -result
  }
  return result, pgam
}

func gamma_incomplete_imp(a, x float64, normalised, invert bool) float64 {

  result := 0.0

  if(int(a) >= MaxFactorial && !normalised) {
    //
    // When we're computing the non-normalized incomplete gamma
    // and a is large the result is rather hard to compute unless
    // we use logs.  There are really two options - if x is a long
    // way from a in value then we can reliably use methods 2 and 4
    // below in logarithmic form and go straight to the result.
    // Otherwise we let the regularized gamma take the strain
    // (the result is unlikely to unerflow in the central region anyway)
    // and combine with lgamma in the hopes that we get a finite result.
    //
    if invert && a*4.0 < x {
      // This is method 4 below, done in logs:
      result  = a*math.Log(x) - x
      result += math.Log(upper_gamma_fraction(a, x))
    } else
    if !invert && a > 4.0*x {
      // This is method 2 below, done in logs:
      result  = a*math.Log(x) - x
      result += math.Log(lower_gamma_series(a, x, 0) / a)
    } else {
      result = gamma_incomplete_imp(a, x, true, invert)
      if result == 0.0 {
        if invert {
          // Try http://functions.wolfram.com/06.06.06.0039.01
          result = 1.0 + 1.0/(12.0*a) + 1.0/(288.0*a*a)
          result = math.Log(result) - a + (a - 0.5)*math.Log(a) + math.Log(M_ROOT_TWO_PI)
        } else {
          // This is method 2 below, done in logs, we're really outside the
          // range of this method, but since the result is almost certainly
          // infinite, we should probably be OK:
          result  = a*math.Log(x) - x
          result += math.Log(lower_gamma_series(a, x, 0)/a)
        }
      } else {
        v, _  := math.Lgamma(a)
        result = math.Log(result) + v
      }
    }
    return math.Exp(result)
   }

  is_int      := false
  is_half_int := false

  if a < 30 && a <= x + 1.0 && x < MaxLogFloat64 {
    fa := math.Floor(a)
    if fa == a {
      is_int = true
    } else
    if math.Abs(fa - a) == 0.5 {
      is_half_int = true
    }
  }

  var eval_method int
   
  if is_int && x > 0.6 {
    // calculate Q via finite sum:
    invert = !invert
    eval_method = 0
  } else
  if is_half_int && x > 0.2  {
    // calculate Q via finite sum for half integer a:
    invert = !invert
    eval_method = 1
  } else
  if x < EpsilonFloat64 && a > 1 {
    eval_method = 6
  } else
  if x < 0.5 {
    //
    // Changeover criterion chosen to give a changeover at Q ~ 0.33
    //
    if -0.4/math.Log(x) < a {
      eval_method = 2
    } else {
      eval_method = 3
    }
  } else
  if x < 1.1 {
    //
    // Changover here occurs when P ~ 0.75 or Q ~ 0.25:
    //
    if x * 0.75 < a {
      eval_method = 2
    } else {
      eval_method = 3
    }
  } else {
    //
    // Begin by testing whether we're in the "bad" zone
    // where the result will be near 0.5 and the usual
     // series and continued fractions are slow to converge:
    //
    use_temme := false
    if normalised &&  a > 20 {
      sigma := math.Abs((x-a)/a)
      if a > 200 && PrecisionFloat64 <= 113 {
        //
        // This limit is chosen so that we use Temme's expansion
        // only if the result would be larger than about 10^-6.
        // Below that the regular series and continued fractions
        // converge OK, and if we use Temme's method we get increasing
        // errors from the dominant erfc term as it's (inexact) argument
        // increases in magnitude.
        //
        if 20/a > sigma*sigma {
          use_temme = true
        }
      } else
      if PrecisionFloat64 <= 64 {
        // Note in this zone we can't use Temme's expansion for 
        // types longer than an 80-bit real:
        // it would require too many terms in the polynomials.
        if sigma < 0.4 {
          use_temme = true
        }
      }
    }
    if use_temme {
      eval_method = 5
    } else {
      //
      // Regular case where the result will not be too close to 0.5.
      //
      // Changeover here occurs at P ~ Q ~ 0.5
      // Note that series computation of P is about x2 faster than continued fraction
      // calculation of Q, so try and use the CF only when really necessary, especially
      // for small x.
      //
      if x - 1.0/(3.0*x) < a {
        eval_method = 2
      } else {
        eval_method = 4
        invert = !invert
      }
    }
  }

  switch eval_method {
  case 0:
    result = finite_gamma_q(a, x)
    if normalised == false {
      result *= math.Gamma(a)
    }
  case 1:
    result = finite_half_gamma_q(a, x)
    if normalised == false {
      result *= math.Gamma(a)
    }
  case 2:
    // Compute P:
    if normalised {
      result = regularised_gamma_prefix(a, x)
    } else {
      result = full_igamma_prefix(a, x)
    }
    if result != 0 {
      //
      // If we're going to be inverting the result then we can
      // reduce the number of series evaluations by quite
      // a few iterations if we set an initial value for the
      // series sum based on what we'll end up subtracting it from
      // at the end.
      // Have to be careful though that this optimization doesn't 
      // lead to spurious numberic overflow.  Note that the
      // scary/expensive overflow checks below are more often
      // than not bypassed in practice for "sensible" input
      // values:
      //
      init_value := 0.0
      optimised_invert := false
      if invert {
        if normalised {
          init_value = 1
        } else {
          init_value = math.Gamma(a)
        }
        if normalised || result >= 1 || math.MaxFloat64*result > init_value {
          init_value /= result
          if normalised || a < 1 || math.MaxFloat64/a > init_value {
            init_value *= -a
            optimised_invert = true
          } else {
            init_value = 0
          }
        } else {
          init_value = 0
        }
      }
      result *= lower_gamma_series(a, x, init_value) / a
      if optimised_invert {
        invert = false
        result = -result
      }
    }
  case 3:
    g := 0.0
    // Compute Q:
    invert = !invert
    result, g = tgamma_small_upper_part(a, x, invert)
    invert = false
    if normalised {
      result /= g
    }
  case 4:
    // Compute Q:
    if normalised {
      result = regularised_gamma_prefix(a, x)
    } else {
      result = full_igamma_prefix(a, x)
    }
    if result != 0 {
      result *= upper_gamma_fraction(a, x)
    }
  case 5:
    result = igamma_temme_large(a, x)
    if x >= a {
      invert = !invert
    }
  case 6:
    // x is so small that P is necessarily very small too,
    // use http://functions.wolfram.com/GammaBetaErf/GammaRegularized/06/01/05/01/01/
    if normalised {
      result = math.Pow(x, a)/a
    } else {
      result = math.Pow(x, a) / math.Gamma(a + 1.0)
    }
    result *= 1.0 - a*x/(a + 1.0)
  }
  if normalised && result > 1.0 {
    result = 1.0
  }
  if invert {
    gam := 1.0
    if !normalised {
      gam = math.Gamma(a)
    }
    result = gam - result
  }

  return result
}

func gamma_p_derivative_imp(a, x float64) float64 {
  //
  // Usual error checks first:
  //
  if a <= 0 {
    return math.NaN()
  }
  if x < 0 {
    return math.NaN()
  }
  //
  // Now special cases:
  //
  if x == 0.0 {
    if a > 1.0 {
      return 0.0
    } else
    if a == 1.0 {
      return 1.0
    } else {
      return math.Inf(1)
    }
  }
  //
  // Normal case:
  //
  f1 := regularised_gamma_prefix(a, x)
  if x < 1.0 && math.MaxFloat64*x < f1 {
    // overflow:
    return math.Inf(1)
   }
   if f1 == 0.0 {
     // Underflow in calculation, use logs instead:
     v, _ := math.Lgamma(a)
     f1 = a*math.Log(x) - x - v - math.Log(x)
     f1 = math.Exp(f1)
   } else {
     f1 /= x
   }
   return f1
}

func gamma_p_second_derivative_imp(a, x float64) float64 {
  t := gamma_p_derivative_imp(a, x)
  return (a-1.0)*t/x - t
}

/* -------------------------------------------------------------------------- */

//
// Full upper incomplete gamma:
//
func GammaUpper(a, z float64) float64 {
  return gamma_incomplete_imp(a, z, false, true)
}

//
// Full lower incomplete gamma:
//
func GammaLower(a, z float64) float64 {
  return gamma_incomplete_imp(a, z, false, false)
}

//
// Regularised lower incomplete gamma:
//
func GammaP(a, z float64) float64 {
  return gamma_incomplete_imp(a, z, true, false)
}

func GammaPfirstDerivative(a, z float64) float64 {
  return gamma_p_derivative_imp(a, z)
}

func GammaPsecondDerivative(a, z float64) float64 {
  return gamma_p_second_derivative_imp(a, z)
}

//
// Regularised upper incomplete gamma:
//
func GammaQ(a, z float64) float64 {
  return gamma_incomplete_imp(a, z, true, true)
}
