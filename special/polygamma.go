/* Copyright (C) 2016 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/detail/polygamma.hpp
 */

//  Copyright 2013 Nikhar Agrawal
//  Copyright 2013 Christopher Kormanyos
//  Copyright 2014 John Maddock
//  Copyright 2013 Paul Bristow
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

package special

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

/*
 * for large values of x, i.e. x > 400
 */

func polygamma_atinfinityplus(n int, x float64) float64 {

  if float64(n) + x == x {
    // x is crazy large, just concentrate on the first part of the expression and use logs:
    if n == 1 {
      return 1.0 / x
    }
    nlx := float64(n) * math.Log(x);
    if nlx < MaxLogFloat64 && n < factorialMax {
      if n & 1 == 1 {
        return  Factorial(n - 1)*math.Pow(x, -float64(n))
      } else {
        return -Factorial(n - 1)*math.Pow(x, -float64(n))
      }
    } else {
      v, _ := math.Lgamma(float64(n))
      if n & 1 == 1 {
        return  math.Exp(v - float64(n)*math.Log(x))
      } else {
        return -math.Exp(v - float64(n)*math.Log(x))
      }
    }
  }
  var term, sum, part_term float64
  x_squared := x*x
  //
  // Start by setting part_term to:
  //
  // (n-1)! / x^(n+1)
  //
  // which is common to both the first term of the series (with k = 1)
  // and to the leading part.  
  // We can then get to the leading term by:
  //
  // part_term * (n + 2 * x) / 2
  //
  // and to the first term in the series 
  // (excluding the Bernoulli number) by:
  //
  // part_term n * (n + 1) / (2x)
  //
  // If either the factorial would overflow,
  // or the power term underflows, this just gets set to 0 and then we
  // know that we have to use logs for the initial terms:
  //
  if n > factorialMax && float64(n)*float64(n) > MaxLogFloat64 {
    part_term = 0.0
  } else {
    part_term = Factorial(n - 1)*math.Pow(x, float64(-n-1))
  }
  if part_term == 0 {
    v, _ := math.Lgamma(float64(n))
    // Either n is very large, or the power term underflows,
    // set the initial values of part_term, term and sum via logs:
    part_term  = v - float64(n + 1)*math.Log(x)
    sum        = math.Exp(part_term + math.Log(float64(n) + 2.0*x) - math.Log(2.0))
    part_term += math.Log(float64(n*(n + 1))) - math.Log(2.0) - math.Log(x)
    part_term  = math.Exp(part_term)
  } else {
    sum        = part_term*(float64(n) + 2.0*x)/2.0;
    part_term *= float64(n*(n + 1))/2.0
    part_term /= x
  }
  //
  // If the leading term is 0, so is the result:
  //
  if sum == 0.0 {
    return sum
  }
  for k := 1;; {
    term = part_term * BernoulliNumber(k)
    sum += term
    //
    // Normal termination condition:
    //
    if math.Abs(term/sum) < EpsilonFloat64 {
      break
    }
    //
    // Increment our counter, and move part_term on to the next value:
    //
    k++
    part_term *= float64((n + 2 * k - 2)*(n - 1 + 2 * k))
    part_term /= float64((2 * k - 1) * 2 * k)
    part_term /= x_squared;
    //
    // Emergency get out termination condition:
    //
    if(k > SeriesIterationsMax) {
      panic("exceeded maximum series evaluations")
    }
  }
  if (n - 1) & 1 == 1 {
    sum = -sum
  }
  return sum;
}

func polygamma_attransitionplus(n int, x float64) float64 {

  // See: http://functions.wolfram.com/GammaBetaErf/PolyGamma2/16/01/01/0017/

  digitsBase10 := float64(PrecisionFloat64 * 301 / 1000)

  // Use N = (0.4 * digits) + (4 * n) for target value for x:
  d4d  := int(0.4 * digitsBase10)

  N    := d4d + (4 * n)
  m    := n
  iter := N - int(math.Trunc(x))

  if iter > SeriesIterationsMax {
    panic("exceeded maximum series evaluations")
  }

  minus_m_minus_one := -m - 1;

  z    := x
  sum0 := 0.0
  z_plus_k_pow_minus_m_minus_one := 0.0

  // Forward recursion to larger x, need to check for overflow first though:
  if math.Log(z + float64(iter)) * float64(minus_m_minus_one) > -MaxLogFloat64 {
    for k := 1; k <= iter; k++ {
      z_plus_k_pow_minus_m_minus_one = math.Pow(z, float64(minus_m_minus_one))
      sum0 += z_plus_k_pow_minus_m_minus_one
      z    += 1.0
    }
    sum0 *= Factorial(n)
  } else {
    for k := 1; k <= iter; k++ {
      v, _ := math.Lgamma(float64(n + 1))
      log_term := math.Log(z) * float64(minus_m_minus_one) + v
      sum0 += math.Exp(log_term)
      z    += 1
    }
  }
  if (n - 1) & 1 == 1 {
    sum0 = -sum0
  }
  return sum0 + polygamma_atinfinityplus(n, z)
}

func polygamma_nearzero(n int, x float64) float64 {
  sum := 0.0
  //
  // If we take this expansion for polygamma: http://functions.wolfram.com/06.15.06.0003.02
  // and substitute in this expression for polygamma(n, 1): http://functions.wolfram.com/06.15.03.0009.01
  // we get an alternating series for polygamma when x is small in terms of zeta functions of
  // integer arguments (which are easy to evaluate, at least when the integer is even).
  //
  // In order to avoid spurious overflow, save the n! term for later, and rescale at the end:
  //
  scale := Factorial(n)
  //
  // "factorial_part" contains everything except the zeta function
  // evaluations in each term:
  //
  factorial_part := 1.0
  //
  // "prefix" is what we'll be adding the accumulated sum to, it will
  // be n! / z^(n+1), but since we're scaling by n! it's just 
  // 1 / z^(n+1) for now:
  //
  prefix := math.Pow(x, float64(n + 1)) // TODO: check overflow
  prefix = 1.0/prefix
  //
  // First term in the series is necessarily < zeta(2) < 2, so
  // ignore the sum if it will have no effect on the result anyway:
  //
  if prefix > 2.0/EpsilonFloat64 {
    if math.MaxFloat64 / prefix < scale {
      goto overflow
    }
    if n & 1 == 1 {
      return prefix * scale
    } else {
      return -prefix * scale
    }
  }
  //
  // As this is an alternating series we could accelerate it using 
  // "Convergence Acceleration of Alternating Series",
  // Henri Cohen, Fernando Rodriguez Villegas, and Don Zagier, Experimental Mathematics, 1999.
  // In practice however, it appears not to make any difference to the number of terms
  // required except in some edge cases which are filtered out anyway before we get here.
  //
  sum = prefix
  for k := 0;; {
    // Get the k'th term:
    term := factorial_part * Zeta(float64(k + n + 1))
    sum  += term
    // Termination condition:
    if math.Abs(term) < math.Abs(sum * EpsilonFloat64) {
      break
    }
    //
    // Move on k and factorial_part:
    //
    k++
    factorial_part *= (-x * float64(n + k)) / float64(k)
    //
    // Last chance exit:
    //
    if(k > SeriesIterationsMax) {
      panic("exceeded maximum series evaluations")
    }
  }
     //
  // We need to multiply by the scale, at each stage checking for oveflow:
  //
  if math.MaxFloat64 / scale < sum {
    goto overflow
  }
  sum *= scale;

  if n & 1 == 1 {
    return sum
  } else {
    return -sum
  }
overflow:
  if n & 1 == 1 {
    // n is odd integer => lim_{t->0} Polygamma(n, t) = +Inf (from below and above)
    return math.Inf(1)
  } else {
    // n is even integer => lim_{t->0} Polygamma(n, t) = +/-Inf (sign depends on direction)
    return math.NaN()
  }
}

func poly_cot_pi(n int, x, xc float64) float64 {
  result := 0.0
  // Return n'th derivative of cot(pi*x) at x, these are simply
  // tabulated for up to n = 9, beyond that it is possible to
  // calculate coefficients as follows:
  //
  // The general form of each derivative is:
  //
  // pi^n * SUM{k=0, n} C[k,n] * cos^k(pi * x) * csc^(n+1)(pi * x)
  //
  // With constant C[0,1] = -1 and all other C[k,n] = 0;
  // Then for each k < n+1:
  // C[k-1, n+1]  -= k * C[k, n];
  // C[k+1, n+1]  += (k-n-1) * C[k, n];
  //
  // Note that there are many different ways of representing this derivative thanks to
  // the many trigomonetric identies available.  In particular, the sum of powers of
  // cosines could be replaced by a sum of cosine multiple angles, and indeed if you
  // plug the derivative into Mathematica this is the form it will give.  The two
  // forms are related via the Chebeshev polynomials of the first kind and
  // T_n(cos(x)) = cos(n x).  The polynomial form has the great advantage that
  // all the cosine terms are zero at half integer arguments - right where this
  // function has it's minumum - thus avoiding cancellation error in this region.
  //
  // And finally, since every other term in the polynomials is zero, we can save
  // space by only storing the non-zero terms.  This greatly complexifies
  // subscripting the tables in the calculation, but halves the storage space
  // (and complexity for that matter).
  //
  s := 0.0
  if math.Abs(x) < math.Abs(xc) {
    s = math.Sin(x*math.Pi)
  } else {
    s = math.Sin(xc*math.Pi)
  }
  c := math.Cos(x*math.Pi)

  switch n {
  case 1:
    return -math.Pi/(s*s)
  case 2:
    return 2.0*math.Pi*math.Pi*c/math.Pow(s, 3.0)
  case 3:
    P := NewEvenPolynomial([]float64{ -2, -4 })
    return math.Pow(math.Pi, 3.0) * P.Eval(c) / math.Pow(s, 4)
  case 4:
    P := NewEvenPolynomial([]float64{ 16, 8 })
    return math.Pow(math.Pi, 4) * c * P.Eval(c) / math.Pow(s, 5)
  case 5:
    P := NewEvenPolynomial([]float64{ -16, -88, -16 })
    return math.Pow(math.Pi, 5) * P.Eval(c) / math.Pow(s, 6)
  case 6:
    P := NewEvenPolynomial([]float64{ 272, 416, 32 })
    return math.Pow(math.Pi, 6) * c * P.Eval(c) / math.Pow(s, 7)
  case 7:
    P := NewEvenPolynomial([]float64{ -272, -2880, -1824, -64 })
    return math.Pow(math.Pi, 7) * P.Eval(c) / math.Pow(s, 8)
  case 8:
    P := NewEvenPolynomial([]float64{ 7936, 24576, 7680, 128 })
    return math.Pow(math.Pi, 8) * c * P.Eval(c) / math.Pow(s, 9)
  case 9:
    P := NewEvenPolynomial([]float64{ -7936, -137216, -185856, -31616, -256 })
    return math.Pow(math.Pi, 9) * P.Eval(c) / math.Pow(s, 10)
  case 10:
    P := NewEvenPolynomial([]float64{ 353792, 1841152, 1304832, 128512, 512 })
    return math.Pow(math.Pi, 10) * c * P.Eval(c) / math.Pow(s, 11)
  case 11:
    P := NewEvenPolynomial([]float64{ -353792, -9061376, -21253376, -8728576, -518656, -1024})
    return math.Pow(math.Pi, 11) * P.Eval(c) / math.Pow(s, 12)
  case 12:
    P := NewEvenPolynomial([]float64{ 22368256, 175627264, 222398464, 56520704, 2084864, 2048 })
    return math.Pow(math.Pi, 12) * c * P.Eval(c) / math.Pow(s, 13)
  case 13:
    P := NewEvenPolynomial([]float64{ -22368256, -795300864, -2868264960, -2174832640, -357888000, -8361984, -4096 })
    return math.Pow(math.Pi, 13) * P.Eval(c) / math.Pow(s, 14)
  case 14:
    P := NewEvenPolynomial([]float64{ 1903757312, 21016670208, 41731645440, 20261765120, 2230947840, 33497088, 8192 })
    return math.Pow(math.Pi, 14) * c * P.Eval(c) / math.Pow(s, 15)
  case 15:
    P := NewEvenPolynomial([]float64{ -1903757312, -89702612992, -460858269696, -559148810240, -182172651520, -13754155008, -134094848, -16384 })
    return math.Pow(math.Pi, 15) * P.Eval(c) / math.Pow(s, 16)
  case 16:
    P := NewEvenPolynomial([]float64{ 209865342976, 3099269660672, 8885192097792, 7048869314560, 1594922762240, 84134068224, 536608768, 32768 })
    return math.Pow(math.Pi, 16) * c * P.Eval(c) / math.Pow(s, 17)
  case 17:
    P := NewEvenPolynomial([]float64{ -209865342976, -12655654469632, -87815735738368, -155964390375424, -84842998005760, -13684856848384, -511780323328, -2146926592, -65536 })
    return math.Pow(math.Pi, 17) * P.Eval(c) / math.Pow(s, 18)
  case 18:
    P := NewEvenPolynomial([]float64{ 29088885112832, 553753414467584, 2165206642589696, 2550316668551168, 985278548541440, 115620218667008, 3100738912256, 8588754944, 131072 })
    return math.Pow(math.Pi, 18) * c * P.Eval(c) / math.Pow(s, 19)
  case 19:
    P := NewEvenPolynomial([]float64{ -29088885112832, -2184860175433728, -19686087844429824, -48165109676113920, -39471306959486976, -11124607890751488, -965271355195392, -18733264797696, -34357248000, -262144 })
    return math.Pow(math.Pi, 19) * P.Eval(c) / math.Pow(s, 20)
  case 20:
    P := NewEvenPolynomial([]float64{ 4951498053124096, 118071834535526400, 603968063567560704, 990081991141490688, 584901762421358592, 122829335169859584, 7984436548730880, 112949304754176, 137433710592, 524288 })
    return math.Pow(math.Pi, 20) * c * P.Eval(c) / math.Pow(s, 21)
  }

  //
  // We'll have to compute the coefficients up to n, 
  // complexity is O(n^2) which we don't worry about for now
  // as the values are computed once and then cached.
  // However, if the final evaluation would have too many
  // terms just bail out right away:
  //
  if n/2 > SeriesIterationsMax {
    panic("n is too large")
  }
  table := [][]float64{{-1.0}}
  index := n - 1

  if index >= len(table) {
    for i := len(table)-1; i < index; i++ {

      offset           := i & 1                                // 1 if the first cos power is 0, otherwise 0.
      sin_order        := i + 2                                // order of the sin term
      max_cos_order    := sin_order - 1                        // largest order of the polynomial of cos terms
      max_columns      := (max_cos_order-offset)/2             // How many entries there are in the current row.
      next_offset      := 0
      next_max_columns := (max_cos_order + 1 - next_offset)/2  // How many entries there will be in the next row

      if offset == 0 {
        next_offset = 1
      }

      table = append(table, make([]float64, next_max_columns + 1))

      for column := 0; column <= max_columns; column++ {
        cos_order := 2 * column + offset;  // order of the cosine term in entry "column"
        if column >= len(table[i]) {
          panic("index out of range")
        }
        if (cos_order + 1)/2 >= len(table[i + 1]) {
          panic("index out of range")
        }
        table[i + 1][(cos_order + 1) / 2] += (float64(cos_order - sin_order) * table[i][column]) / float64(sin_order - 1)

        if cos_order != 0 {
          table[i + 1][(cos_order - 1) / 2] += (-float64(cos_order) * table[i][column]) / float64(sin_order - 1)
        }
      }
    }
  }
  P   := NewEvenPolynomial(table[index])
  sum := P.Eval(c)

  if index & 1 == 1 {
    sum *= c  // First coeffient is order 1, and really an odd polynomial.
  }
  if sum == 0 {
    return sum
  }
  //
  // The remaining terms are computed using logs since the powers and factorials
  // get real large real quick:
  //
  power_terms := float64(n) * math.Log(math.Pi)
  v, _        := math.Lgamma(float64(n))
  if s == 0 {
    goto overflow
  }
  power_terms -= math.Log(math.Abs(s)) * float64(n + 1)
  power_terms += v
  power_terms += math.Log(math.Abs(sum))

  if power_terms > MaxLogFloat64 {
    goto overflow
  }
  result = math.Exp(power_terms)

  if math.Signbit(sum) {
    result *= -1
  }
  if (s < 0) && ((n + 1) & 1 == 1) {
    result *= -1
  }
  return result
overflow:
  if n & 1 == 1 {
    // n is odd integer => lim_{t->x} Polygamma(n, t) = -Inf (from below and above)
    return math.Inf(-1)
  } else {
    // n is even integer => lim_{t->x} Polygamma(n, t) = +/-Inf (sign depends on direction)
    return math.NaN()
  }
}

func polygamma_imp(n int, x float64) float64 {
  digitsBase10 := float64(PrecisionFloat64 * 301 / 1000)

  if(n < 0) {
    panic("order must be >= 0")
  }
  if(x < 0.0) {
    if(math.Floor(x) == x) {
      if n & 1 == 1 {
        // n is odd integer => lim_{t->x} Polygamma(n, t) = +Inf (from below and above)
        return math.Inf(1)
      } else {
        // n is even integer => lim_{t->x} Polygamma(n, t) = +/-Inf (sign depends on direction)
        return math.NaN()
      }
    }
    z      := 1.0 - x
    result := polygamma_imp(n, z) + math.Pi * poly_cot_pi(n, z, x)

    if n & 1 == 1 {
      return -result
    } else {
      return  result
    }
  }
  //
  // Limit for use of small-x-series is chosen
  // so that the series doesn't go too divergent
  // in the first few terms.  Ordinarily this
  // would mean setting the limit to ~ 1 / n,
  // but we can tolerate a small amount of divergence:
  //
  small_x_limit := math.Min(5.0 / float64(n), 0.25)
  if x < small_x_limit {
    return polygamma_nearzero(n, x)
  } else if x > 0.4 * digitsBase10 + 4.0 * float64(n) {
    return polygamma_atinfinityplus(n, x)
  } else if(x == 1) {
    if n & 1 == 1 {
      return  Factorial(n) * Zeta(float64(n + 1))
    } else {
      return -Factorial(n) * Zeta(float64(n + 1))
    }
  } else if x == 0.5 {
    result := Factorial(n) * Zeta(float64(n + 1))
    if n & 1 == 0 {
      result = -result
    }
    if math.Abs(result) >= math.MaxFloat64*math.Pow(2.0, float64(-n-1)) {
      return math.Inf(-1)
    }
    result *= math.Pow(2.0,  float64(n + 1)) - 1.0

    return result
  } else {
    return polygamma_attransitionplus(n, x)
  }
}

/* -------------------------------------------------------------------------- */

func Polygamma(n int, x float64) float64 {
  if n == 0 {
    return Digamma(x)
  } else if n == 1 {
    return Trigamma(x)
  } else {
    return polygamma_imp(n, x)
  }
}
