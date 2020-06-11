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

const need_i = 1
const need_k = 2

/* -------------------------------------------------------------------------- */

func iround(x float64) int {
	if x < 0 {
		return int(x - 0.5)
	} else {
		return int(x + 0.5)
	}
}

/* -------------------------------------------------------------------------- */

func bessel_i0(x float64) float64 {
	if x < 7.75 {
		// Bessel I0 over[10 ^ -16, 7.75]
		// Max error in interpolated form : 3.042e-18
		// Max Error found at double precision = Poly : 5.106609e-16 Cheb : 5.239199e-16
		P := NewPolynomial([]float64{
			1.00000000000000000e+00,
			2.49999999999999909e-01,
			2.77777777777782257e-02,
			1.73611111111023792e-03,
			6.94444444453352521e-05,
			1.92901234513219920e-06,
			3.93675991102510739e-08,
			6.15118672704439289e-10,
			7.59407002058973446e-12,
			7.59389793369836367e-14,
			6.27767773636292611e-16,
			4.34709704153272287e-18,
			2.63417742690109154e-20,
			1.13943037744822825e-22,
			9.07926920085624812e-25})
		a := x * x / 4
		return a*P.Eval(a) + 1.0
	} else if x < 500 {
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
			2.17587543863819074e+15})
		return math.Exp(x) * P.Eval(1.0/x) / math.Sqrt(x)
	} else {
		// Max error in interpolated form : 2.437e-18
		// Max Error found at double precision = Poly : 1.216719e-16
		P := NewPolynomial([]float64{
			3.98942280401432905e-01,
			4.98677850491434560e-02,
			2.80506308916506102e-02,
			2.92179096853915176e-02,
			4.53371208762579442e-02})
		ex := math.Exp(x / 2)
		result := ex * P.Eval(1.0/x) / math.Sqrt(x)
		result *= ex
		return result
	}
}

/* -------------------------------------------------------------------------- */

func bessel_i1(x float64) float64 {
	if x < 7.75 {
		// Bessel I0 over[10 ^ -16, 7.75]
		// Max error in interpolated form: 5.639e-17
		// Max Error found at double precision = Poly: 1.795559e-16
		P := NewPolynomial([]float64{
			8.333333333333333803e-02,
			6.944444444444341983e-03,
			3.472222222225921045e-04,
			1.157407407354987232e-05,
			2.755731926254790268e-07,
			4.920949692800671435e-09,
			6.834657311305621830e-11,
			7.593969849687574339e-13,
			6.904822652741917551e-15,
			5.220157095351373194e-17,
			3.410720494727771276e-19,
			1.625212890947171108e-21,
			1.332898928162290861e-23})
		a := x * x / 4
		Q := NewPolynomial([]float64{
			1, 0.5, P.Eval(a)})
		return x * Q.Eval(a) / 2
	} else if x < 500 {
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
			-2.213318202179221945e+15})
		return math.Exp(x) * P.Eval(1/x) / math.Sqrt(x)
	} else {
		// Max error in interpolated form: 1.320e-19
		// Max Error found at double precision = Poly: 7.065357e-17
		P := NewPolynomial([]float64{
			3.989422804014314820e-01,
			-1.496033551467584157e-01,
			-4.675105322571775911e-02,
			-4.090421597376992892e-02,
			-5.843630344778927582e-02})
		ex := math.Exp(x / 2)
		result := ex * P.Eval(1/x) / math.Sqrt(x)
		result *= ex
		return result
	}
}

/* -------------------------------------------------------------------------- */

type cyl_bessel_i_small_z struct {
	k    int
	v    float64
	term float64
	mult float64
}

func new_cyl_bessel_i_small_z(v, z float64) *cyl_bessel_i_small_z {
	r := cyl_bessel_i_small_z{}
	r.term = 1
	r.k = 0
	r.v = v
	r.mult = z * z / 4
	return &r
}

func (obj *cyl_bessel_i_small_z) Eval() float64 {
	r := obj.term
	obj.k += 1
	obj.term *= obj.mult / float64(obj.k)
	obj.term /= float64(obj.k) + obj.v
	return r
}

/* -------------------------------------------------------------------------- */

func bessel_i_small_z_series(v, x float64) float64 {
	var prefix float64

	if v < float64(MaxFactorial) {
		prefix = math.Pow(x/2.0, v) / math.Gamma(v+1)
	} else {
		t, _ := math.Lgamma(v + 1)
		prefix = math.Log(x/2)*v - t
		prefix = math.Exp(prefix)
	}
	if prefix == 0.0 {
		return prefix
	}

	s := new_cyl_bessel_i_small_z(v, x)

	return prefix * SumSeries(s, 0.0, 2.22045e-16, SeriesIterationsMax)
}

/* -------------------------------------------------------------------------- */

func asymptotic_bessel_i_large_x(v, x float64) float64 {
	s := 1.0
	mu := 4.0 * v * v
	ex := 8.0 * x
	num := mu - 1.0
	denom := ex

	s -= num / denom

	num *= mu - 9
	denom *= ex * 2
	s += num / denom

	num *= mu - 25
	denom *= ex * 3
	s -= num / denom

	e := math.Exp(x / 2)

	s = e * (e * s / math.Sqrt(2.0*x*math.Pi))

	return s
}

/* -------------------------------------------------------------------------- */

func temme_ik(v, x float64) (float64, float64) {
	var K, K1, f, h, p, q, coef, sum, sum1, tolerance float64
	var a, b, c, d, sigma, gamma1, gamma2 float64

	// |x| <= 2, Temme series converge rapidly
	// |x| > 2, the larger the |x|, the slower the convergence
	if math.Abs(x) > 2 {
		panic("internal error")
	}
	if math.Abs(v) > 0.5 {
		panic("internal error")
	}

	gp := tgamma1pm1(v)
	gm := tgamma1pm1(-v)

	a = math.Log(x / 2)
	b = math.Exp(v * a)
	sigma = -a * v
	if math.Abs(v) < EpsilonFloat64 {
		c = 1.0
	} else {
		c = SinPi(v) / (v * math.Pi)
	}
	if math.Abs(sigma) < EpsilonFloat64 {
		d = 1.0
	} else {
		d = math.Sinh(sigma) / sigma
	}
	if math.Abs(v) < EpsilonFloat64 {
		gamma1 = -M_EULER
	} else {
		gamma1 = (0.5 / v) * (gp - gm) * c
	}
	gamma2 = (2 + gp + gm) * c / 2

	// initial values
	p = (gp + 1) / (2 * b)
	q = (1 + gm) * b / 2
	f = (math.Cosh(sigma)*gamma1 + d*(-a)*gamma2) / c
	h = p
	coef = 1
	sum = coef * f
	sum1 = coef * h

	// series summation
	tolerance = EpsilonFloat64
	for k := 1; k < SeriesIterationsMax; k++ {
		kf := float64(k)
		f = (kf*f + p + q) / (kf*kf - v*v)
		p /= kf - v
		q /= kf + v
		h = p - kf*f
		coef *= x * x / (4 * kf)
		sum += coef * f
		sum1 += coef * h
		if math.Abs(coef*f) < math.Abs(sum)*tolerance {
			break
		}
	}

	K = sum
	K1 = 2 * sum1 / x

	return K, K1
}

/* -------------------------------------------------------------------------- */

// Evaluate continued fraction fv = I_(v+1) / I_v, derived from
// Abramowitz and Stegun, Handbook of Mathematical Functions, 1972, 9.1.73
func CF1_ik(v, x float64) float64 {
	var C, D, f, a, b, delta, tiny, tolerance float64

	// |x| <= |v|, CF1_ik converges rapidly
	// |x| > |v|, CF1_ik needs O(|x|) iterations to converge

	// modified Lentz's method, see
	// Lentz, Applied Optics, vol 15, 668 (1976)
	tolerance = 2.0 * EpsilonFloat64
	tiny = math.Sqrt(math.SmallestNonzeroFloat64)
	C = tiny
	f = tiny // b0 = 0, replace with tiny
	D = 0
	for k := 1; k < SeriesIterationsMax; k++ {
		a = 1
		b = 2 * (v + float64(k)) / x
		C = b + a/C
		D = b + a*D
		if C == 0.0 {
			C = tiny
		}
		if D == 0.0 {
			D = tiny
		}
		D = 1 / D
		delta = C * D
		f *= delta
		if math.Abs(delta-1.0) <= tolerance {
			break
		}
	}

	return f
}

/* -------------------------------------------------------------------------- */

// Calculate K(v, x) and K(v+1, x) by evaluating continued fraction
// z1 / z0 = U(v+1.5, 2v+1, 2x) / U(v+0.5, 2v+1, 2x), see
// Thompson and Barnett, Computer Physics Communications, vol 47, 245 (1987)
func CF2_ik(v, x float64) (float64, float64) {
	var Kv, Kv1, S, C, Q, D, f, a, b, q, delta, tolerance, current, prev float64

	// |x| >= |v|, CF2_ik converges rapidly
	// |x| -> 0, CF2_ik fails to converge

	if math.Abs(x) <= 1 {
		panic("internal error")
	}

	// Steed's algorithm, see Thompson and Barnett,
	// Journal of Computational Physics, vol 64, 490 (1986)
	tolerance = EpsilonFloat64
	a = v*v - 0.25
	b = 2 * (x + 1) // b1
	D = 1 / b       // D1 = 1 / b1
	f = D
	delta = D   // f1 = delta1 = D1, coincidence
	prev = 0    // q0
	current = 1 // q1
	Q = -a
	C = -a          // Q1 = C1 because q1 = 1
	S = 1 + Q*delta // S1

	for k := 2; k < SeriesIterationsMax; k++ { // starting from 2
		// continued fraction f = z1 / z0
		a -= 2 * (float64(k) - 1)
		b += 2
		D = 1 / (b + a*D)
		delta *= b*D - 1
		f += delta

		// series summation S = 1 + \sum_{n=1}^{\infty} C_n * z_n / z_0
		q = (prev - (b-2)*current) / a
		prev = current
		current = q // forward recurrence for q
		C *= -a / float64(k)
		Q += C * q
		S += Q * delta
		//
		// Under some circumstances q can grow very small and C very
		// large, leading to under/overflow.  This is particularly an
		// issue for types which have many digits precision but a narrow
		// exponent range.  A typical example being a "double double" type.
		// To avoid this situation we can normalise q (and related prev/current)
		// and C.  All other variables remain unchanged in value.  A typical
		// test case occurs when x is close to 2, for example cyl_bessel_k(9.125, 2.125).
		//
		if q < EpsilonFloat64 {
			C *= q
			prev /= q
			current /= q
			q = 1
		}

		// S converges slower than f
		if math.Abs(Q*delta) < math.Abs(S)*tolerance {
			break
		}
	}

	if x >= MaxLogFloat64 {
		Kv = math.Exp(0.5*math.Log(math.Pi/(2.0*x)) - x - math.Log(S))
	} else {
		Kv = math.Sqrt(math.Pi/(2.0*x)) * math.Exp(-x) / S
	}
	Kv1 = Kv * (0.5 + v + x + (v*v-0.25)*f) / x

	return Kv, Kv1
}

/* -------------------------------------------------------------------------- */

// Compute I(v, x) and K(v, x) simultaneously by Temme's method, see
// Temme, Journal of Computational Physics, vol 19, 324 (1975)
func bessel_ik(v, x float64, kind int) (float64, float64) {
	// Kv1 = K_(v+1), fv = I_(v+1) / I_v
	// Ku1 = K_(u+1), fu = I_(u+1) / I_u
	var u, I, K, Iv, Kv, Kv1, Ku, Ku1, fv float64
	var W, current, prev, next float64
	var n, k int

	reflect := false
	org_kind := kind

	if v < 0 {
		reflect = true
		v = -v // v is non-negative from here
		kind |= need_k
	}
	n = iround(v)
	u = v - float64(n) // -1/2 <= u < 1/2

	if x < 0 {
		panic(fmt.Sprintf("Got x = %f but real argument x must be non-negative, complex number result not supported.", x))
	}
	if x == 0.0 {
		if v == 0.0 {
			I = 1.0
		} else {
			I = 0.0
		}
		if kind&need_k != 0 {
			K = math.Inf(1)
		} else {
			K = math.NaN() // any value will do
		}
		if reflect && (kind&need_i) != 0 {
			z := u + float64(n%2)
			if SinPi(z) != 0.0 {
				I = math.Inf(1)
			}
		}
		return I, K
	}
	// x is positive until reflection
	W = 1.0 / x // Wronskian
	if x <= 2 { // x in (0, 2]
		Ku, Ku1 = temme_ik(u, x) // Temme series
	} else { // x in (2, \infty)
		Ku, Ku1 = CF2_ik(u, x) // continued fraction CF2_ik
	}
	prev = Ku
	current = Ku1
	scale := 1.0
	scale_sign := 1.0
	for k = 1; k <= n; k++ { // forward recurrence for K
		fact := 2.0 * (u + float64(k)) / x
		if (math.MaxFloat64-math.Abs(prev))/fact < math.Abs(current) {
			prev /= current
			scale /= current
			if current < 0 {
				scale_sign *= -1.0
			}
			current = 1
		}
		next = fact*current + prev
		prev = current
		current = next
	}
	Kv = prev
	Kv1 = current

	if kind&need_i != 0 {
		lim := (4.0*v*v + 10.0) / (8.0 * x)
		lim *= lim
		lim *= lim
		lim /= 24.0
		if (lim < EpsilonFloat64*10.0) && (x > 100.0) {
			// x is huge compared to v, CF1 may be very slow
			// to converge so use asymptotic expansion for large
			// x case instead.  Note that the asymptotic expansion
			// isn't very accurate - so it's deliberately very hard
			// to get here - probably we're going to overflow:
			Iv = asymptotic_bessel_i_large_x(v, x)
		} else if (v > 0.0) && (x/v < 0.25) {
			Iv = bessel_i_small_z_series(v, x)
		} else {
			fv = CF1_ik(v, x)              // continued fraction CF1_ik
			Iv = scale * W / (Kv*fv + Kv1) // Wronskian relation
		}
	} else {
		Iv = math.NaN() // any value will do
	}

	if reflect {
		z := u + float64(n%2)
		fact := 2.0 / math.Pi * SinPi(z) * Kv
		if fact == 0 {
			I = Iv
		} else if math.MaxFloat64*scale < fact {
			if org_kind&need_i != 0 {
				if fact*scale_sign < 0 {
					I = math.Inf(-1)
				} else {
					I = math.Inf(1)
				}
			} else {
				I = 0.0
			}
		} else {
			I = Iv + fact/scale // reflection formula
		}
	} else {
		I = Iv
	}
	if math.MaxFloat64*scale < Kv {
		if org_kind&need_k != 0 {
			if Kv*scale_sign < 0.0 {
				K = math.Inf(-1)
			} else {
				K = math.Inf(1)
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

func bessel_i_imp(v, x float64) float64 {
	//
	// This handles all the bessel I functions, note that we don't optimise
	// for integer v, other than the v = 0 or 1 special cases, as Millers
	// algorithm is at least as inefficient as the general case (the general
	// case has better error handling too).
	//
	if x < 0 {
		// better have integer v:
		if math.Floor(v) == v {
			r := bessel_i_imp(v, -x)
			if iround(v)&1 != 0 {
				return -r
			} else {
				return r
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
		if x >= MaxLogFloat64 {
			e := math.Exp(x / 2.0)
			return e * (e / math.Sqrt(2.0*x*math.Pi))
		} else {
			return math.Sqrt(2.0/(x*math.Pi)) * math.Sinh(x)
		}
	}
	if v == 0 {
		return bessel_i0(x)
	}
	if v == 1 {
		return bessel_i1(x)
	}
	if v > 0 && x/v < 0.25 {
		return bessel_i_small_z_series(v, x)
	}
	I, _ := bessel_ik(v, x, need_i)
	return I
}

/* -------------------------------------------------------------------------- */

// modified bessel function of the first kind
func BesselI(v, x float64) float64 {
	return bessel_i_imp(v, x)
}
