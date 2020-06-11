/* Copyright (C) 2017 Philipp Benner
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

package vectorEstimator

/* -------------------------------------------------------------------------- */

//import   "fmt"
//import   "os"
import "math"
import "testing"

import . "github.com/pbenner/autodiff/statistics"
import "github.com/pbenner/autodiff/statistics/scalarDistribution"
import "github.com/pbenner/autodiff/statistics/vectorDistribution"
import "github.com/pbenner/autodiff/statistics/scalarEstimator"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/simple"
import "github.com/pbenner/autodiff/algorithm/bfgs"
import "github.com/pbenner/autodiff/algorithm/newton"

import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func TestHmm1(t *testing.T) {
	// Hmm definition
	//////////////////////////////////////////////////////////////////////////////
	pi := NewVector(RealType, []float64{0.6, 0.4})
	tr := NewMatrix(RealType, 2, 2,
		[]float64{0.7, 0.3, 0.4, 0.6})

	e1, _ := scalarEstimator.NewCategoricalEstimator([]float64{0.1, 0.9})
	e2, _ := scalarEstimator.NewCategoricalEstimator([]float64{0.7, 0.3})

	// test Baum-Welch algorithm
	//////////////////////////////////////////////////////////////////////////////
	if estimator, err := NewHmmEstimator(pi, tr, nil, nil, nil, []ScalarEstimator{e1, e2}, 1e-8, -1); err != nil {
		t.Error(err)
	} else {
		hmm1, _ := estimator.GetEstimate()
		x := NewVector(RealType, []float64{1, 1, 1, 1, 1, 1, 0, 0, 1, 0})

		if err := estimator.EstimateOnData([]ConstVector{x}, nil, ThreadPool{}); err != nil {
			t.Error(err)
		} else {
			hmm2, _ := estimator.GetEstimate()

			p1 := NullReal()
			hmm1.LogPdf(p1, x)
			p2 := NullReal()
			hmm2.LogPdf(p2, x)

			if p1.Greater(p2) {
				t.Errorf("Baum-Welch test failed")
			}
			if math.Abs(p2.GetValue() - -4.493268e+00) > 1e-4 {
				t.Errorf("Baum-Welch test failed")
			}
		}
	}
	// test Baum-Welch algorithm with conditioning
	//////////////////////////////////////////////////////////////////////////////
	if estimator, err := NewHmmEstimator(pi, tr, nil, []int{0}, []int{0}, []ScalarEstimator{e1, e2}, 1e-8, -1); err != nil {
		t.Error(err)
	} else {
		hmm1, _ := estimator.GetEstimate()
		x := NewVector(RealType, []float64{1, 1, 1, 1, 1, 1, 0, 0, 1, 0})

		if err := estimator.EstimateOnData([]ConstVector{x}, nil, ThreadPool{}); err != nil {
			t.Error(err)
		} else {
			hmm2, _ := estimator.GetEstimate()

			p1 := NullReal()
			hmm1.LogPdf(p1, x)
			p2 := NullReal()
			hmm2.LogPdf(p2, x)
			if p1.Greater(p2) {
				t.Errorf("Baum-Welch test failed")
			}
			if math.Abs(p2.GetValue() - -5.834855e+00) > 1e-4 {
				t.Errorf("Baum-Welch test failed")
			}
		}
	}
}

func TestHmm2(t *testing.T) {
	// Hmm definition
	//////////////////////////////////////////////////////////////////////////////
	tr := NewMatrix(RealType, 2, 2,
		[]float64{0.7, 0.3, 0.4, 0.6})

	c1, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c2, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	edist := []ScalarPdf{c1, c2}

	pi := NewVector(RealType, []float64{0.6, 0.4})

	x := NewVector(RealType, []float64{1, 1, 1, 1, 1, 1, 0, 0, 1, 0})
	r := NewReal(0.0)

	hmm, err := vectorDistribution.NewHmm(pi, tr, nil, edist)
	if err != nil {
		t.Error(err)
	}
	hmm.SetStartStates([]int{0})
	hmm.SetFinalStates([]int{0})

	penalty := func(p1, p2, c Scalar) Scalar {
		r := NewReal(0.0)
		r.Add(p1, p2)
		r.Sub(r, NewReal(1.0))
		r.Pow(r, NewReal(2.0))
		r.Mul(r, c)
		return r
	}
	objective_template := func(variables Vector, c Scalar) (Scalar, error) {
		// create a new initial normal distribution
		pi := NullVector(RealType, 2)
		tr := NullMatrix(RealType, 2, 2)
		// copy the variables
		pi.At(0).SetValue(1.0)
		pi.At(1).SetValue(1.0)
		tr.At(0, 0).Exp(variables.At(0))
		tr.At(0, 1).Exp(variables.At(1))
		tr.At(1, 0).Exp(variables.At(2))
		tr.At(1, 1).Exp(variables.At(3))
		// construct new Hmm
		hmm, _ := vectorDistribution.NewHmm(pi, tr, nil, edist)
		hmm.SetStartStates([]int{0})
		hmm.SetFinalStates([]int{0})
		// compute objective function
		result := NewScalar(RealType, 0.0)
		// density function
		hmm.LogPdf(r, x)
		result.Add(result, r)
		result.Neg(result)
		// penalty function
		result.Add(result, penalty(pi.At(0), pi.At(1), c))
		result.Add(result, penalty(tr.At(0, 0), tr.At(0, 1), c))
		result.Add(result, penalty(tr.At(1, 0), tr.At(1, 1), c))
		return result, nil
	}
	// hook_bfgs := func(variables, gradient Vector, s Scalar) bool {
	//   fmt.Println("variables:", variables)
	//   fmt.Println("gradient :", gradient)
	//   fmt.Println("y        :", s)
	//   fmt.Println("")
	//   return false
	// }
	// initial value
	vn := hmm.GetParameters()
	vn = vn.Slice(2, vn.Dim())
	// initial penalty strength
	c := NewReal(2.0)
	// run rprop
	for i := 0; i < 20; i++ {
		objective := func(variables Vector) (Scalar, error) {
			return objective_template(variables, c)
		}
		vn, _ = bfgs.Run(objective, vn,
			//bfgs.Hook{hook_bfgs},
			bfgs.Epsilon{1e-8})
		// increase penalty strength
		c.Mul(c, NewReal(2.0))
	}
	// check result
	if math.Abs(Exp(vn.At(0)).GetValue()-8.257028e-01) > 1e-3 ||
		math.Abs(Exp(vn.At(1)).GetValue()-1.743001e-01) > 1e-3 ||
		math.Abs(Exp(vn.At(2)).GetValue()-3.597875e-01) > 1e-3 ||
		math.Abs(Exp(vn.At(3)).GetValue()-6.402134e-01) > 1e-3 {
		t.Error("Hmm test failed!")
	}
}

func TestHmm3(t *testing.T) {
	// Hmm definition
	//////////////////////////////////////////////////////////////////////////////
	tr := NewMatrix(RealType, 2, 2,
		[]float64{5, 1, 1, 10})

	c1, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c2, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	edist := []ScalarPdf{c1, c2}

	pi := NewVector(RealType, []float64{0.6, 0.4})

	x := NewVector(RealType, []float64{1, 1, 1, 1, 1, 1, 0, 0, 1, 0})
	r := NewReal(0.0)

	hmm, err := vectorDistribution.NewHmm(pi, tr, nil, edist)
	if err != nil {
		t.Error(err)
	}

	constraint := func(p1, p2, lambda Scalar) Scalar {
		r := NewReal(0.0)
		r.Add(p1, p2)
		r.Sub(r, NewReal(1.0))
		r.Mul(r, lambda)
		return r
	}
	objective := func(variables Vector) (Scalar, error) {
		// create a new initial normal distribution
		tr := NullMatrix(RealType, 2, 2)
		// copy the variables
		tr.At(0, 0).Exp(variables.At(0))
		tr.At(0, 1).Exp(variables.At(1))
		tr.At(1, 0).Exp(variables.At(2))
		tr.At(1, 1).Exp(variables.At(3))
		// lambda parameters of the Lagrangian
		lambda := variables.Slice(4, 6)
		// construct new Hmm
		hmm, _ := vectorDistribution.NewHmm(pi, tr, nil, edist)
		// compute objective function
		result := NewScalar(RealType, 0.0)
		// density function
		hmm.LogPdf(r, x)
		result.Add(result, r)
		result.Neg(result)
		// constraints
		result.Add(result, constraint(tr.At(0, 0), tr.At(0, 1), lambda.At(0)))
		result.Add(result, constraint(tr.At(1, 0), tr.At(1, 1), lambda.At(1)))
		return result, nil
	}
	// hook_newton := func(x Vector, hessian Matrix, gradient Vector) bool {
	//   fmt.Println("hessian :", hessian)
	//   fmt.Println("gradient:", gradient)
	//   fmt.Println("x       :", x)
	//   fmt.Println("")
	//   return false
	// }
	// initial value
	vn := hmm.GetParameters()
	// drop pi and parameters from the emission distributions
	vn = vn.Slice(2, 6)
	// append Lagriangian lambda parameters
	vn = vn.AppendScalar(NewReal(1.0), NewReal(1.0))
	// find critical points of the Lagrangian
	vn, err = newton.RunCrit(objective, vn,
		//    newton.HookCrit{hook_newton},
		newton.Epsilon{1e-4})
	if err != nil {
		t.Error(err)
	} else {
		// check result
		if math.Abs(Exp(vn.At(0)).GetValue()-8.230221e-01) > 1e-4 ||
			math.Abs(Exp(vn.At(1)).GetValue()-1.769779e-01) > 1e-4 ||
			math.Abs(Exp(vn.At(2)).GetValue()-7.975104e-09) > 1e-4 ||
			math.Abs(Exp(vn.At(3)).GetValue()-1.000000e+00) > 1e-4 {
			t.Error("Hmm test failed!")
		}
	}
}

func TestHmm4(t *testing.T) {
	// Hmm definition
	//////////////////////////////////////////////////////////////////////////////
	pi := NewVector(RealType, []float64{0.6, 0.4})

	tr := NewMatrix(RealType, 2, 2,
		[]float64{0.7, 0.3, 0.4, 0.6})

	c1, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.1, 0.9}))
	c2, _ := scalarDistribution.NewCategoricalDistribution(
		NewVector(RealType, []float64{0.7, 0.3}))
	edist := []ScalarPdf{c1, c2}

	x := NewVector(RealType, []float64{1, 1, 1, 1, 1, 1, 0, 0, 1, 0})
	r := NewReal(0.0)

	hmm, err := vectorDistribution.NewHmm(pi, tr, nil, edist)
	if err != nil {
		t.Error(err)
	}
	hmm.SetStartStates([]int{0})
	hmm.SetFinalStates([]int{0})

	constraint := func(p1, p2, lambda Scalar) Scalar {
		r := NewReal(0.0)
		r.Add(p1, p2)
		r.Sub(r, NewReal(1.0))
		r.Mul(r, lambda)
		return r
	}
	objective := func(variables Vector) (Scalar, error) {
		// create a new initial normal distribution
		tr := NullMatrix(RealType, 2, 2)
		// copy the variables
		tr.At(0, 0).Exp(variables.At(0))
		tr.At(0, 1).Exp(variables.At(1))
		tr.At(1, 0).Exp(variables.At(2))
		tr.At(1, 1).Exp(variables.At(3))
		// lambda parameters of the Lagrangian
		lambda := variables.Slice(4, 6)
		// construct new Hmm
		hmm, _ := vectorDistribution.NewHmm(pi, tr, nil, edist)
		hmm.SetStartStates([]int{0})
		hmm.SetFinalStates([]int{0})
		// compute objective function
		result := NewScalar(RealType, 0.0)
		// density function
		hmm.LogPdf(r, x)
		result.Add(result, r)
		result.Neg(result)
		// constraints
		result.Add(result, constraint(tr.At(0, 0), tr.At(0, 1), lambda.At(0)))
		result.Add(result, constraint(tr.At(1, 0), tr.At(1, 1), lambda.At(1)))
		return result, nil
	}
	// hook_newton := func(x Vector, hessian Matrix, gradient Vector) bool {
	//   fmt.Println("hessian :", hessian)
	//   fmt.Println("gradient:", gradient)
	//   fmt.Println("x       :", x)
	//   fmt.Println("")
	//   return false
	// }
	// initial value
	vn := hmm.GetParameters()
	// drop pi and parameters from the emission distributions
	vn = vn.Slice(2, 6)
	// append Lagriangian lambda parameters
	vn = vn.AppendScalar(NewReal(1.0), NewReal(1.0))
	// run rprop
	vn, err = newton.RunCrit(objective, vn,
		//newton.HookCrit{hook_newton},
		newton.Epsilon{1e-10})
	if err != nil {
		t.Error(err)
	} else {
		// check result
		if math.Abs(Exp(vn.At(0)).GetValue()-8.257028e-01) > 1e-3 ||
			math.Abs(Exp(vn.At(1)).GetValue()-1.743001e-01) > 1e-3 ||
			math.Abs(Exp(vn.At(2)).GetValue()-3.597875e-01) > 1e-3 ||
			math.Abs(Exp(vn.At(3)).GetValue()-6.402134e-01) > 1e-3 {
			t.Error("Hmm test failed!")
		}
	}
}

func TestHmm5(t *testing.T) {

	pi := NewVector(RealType, []float64{0.6, 0.4})

	var tr Matrix = NewMatrix(RealType, 2, 2,
		[]float64{0.7, 0.3, 0.4, 0.6})

	c1, _ := scalarDistribution.NewGammaDistribution(NewReal(0.5), NewReal(2.0))
	c2, _ := scalarDistribution.NewGammaDistribution(NewReal(10.0), NewReal(2.0))

	e1, _ := scalarEstimator.NewNumericEstimator(c1)
	e2, _ := scalarEstimator.NewNumericEstimator(c2)

	e1.Epsilon = 1e-7
	e2.Epsilon = 1e-7

	x := []ConstVector{
		NewVector(RealType, []float64{0.23092451, 0.23092451, 0.23092451, 5.975650, 5.975650, 5.975650}),
		NewVector(RealType, []float64{1.15626248, 1.15626248, 1.15626248, 3.074001, 3.074001, 3.074001}),
		NewVector(RealType, []float64{0.39937995, 0.39937995, 0.39937995, 3.806467, 3.806467, 3.806467}),
		NewVector(RealType, []float64{0.51252240, 0.51252240, 0.51252240, 6.654319, 6.654319, 6.654319}),
		NewVector(RealType, []float64{2.35671304, 2.35671304, 2.35671304, 2.904598, 2.904598, 2.904598}),
		NewVector(RealType, []float64{0.18067285, 0.18067285, 0.18067285, 2.895080, 2.895080, 2.895080}),
		NewVector(RealType, []float64{0.06068149, 0.06068149, 0.06068149, 3.088718, 3.088718, 3.088718}),
		NewVector(RealType, []float64{1.71700325, 1.71700325, 1.71700325, 4.068132, 4.068132, 4.068132}),
		NewVector(RealType, []float64{0.06229591, 0.06229591, 0.06229591, 4.466460, 4.466460, 4.466460}),
		NewVector(RealType, []float64{0.43543498, 0.43543498, 0.43543498, 6.193897, 6.193897, 6.193897})}

	estimator, err := NewHmmEstimator(pi, tr, nil, nil, nil, []ScalarEstimator{e1, e2}, 1e-10, -1)
	if err != nil {
		t.Error(err)
		return
	}
	if err := estimator.EstimateOnData(x, nil, ThreadPool{}); err != nil {
		t.Error(err)
		return
	}
	hmt, _ := estimator.GetEstimate()
	hmm := hmt.(*vectorDistribution.Hmm)

	// correct values
	qi := NewVector(RealType, []float64{7.249908e-01, 2.750092e-01})
	sr := NewMatrix(RealType, 2, 2, []float64{
		6.592349e-01, 3.407651e-01,
		0.000000e+00, 1.000000e+00})

	pi = hmm.Pi
	pi.Map(func(a Scalar) { a.Exp(a) })
	tr = hmm.Tr
	tr.Map(func(a Scalar) { a.Exp(a) })

	if Vnorm(VsubV(pi, qi)).GetValue() > 1e-3 {
		t.Error("Hmm test failed!")
	}
	if Mnorm(MsubM(tr, sr)).GetValue() > 1e-4 {
		t.Error("Hmm test failed!")
	}
	if math.Abs(hmm.Edist[0].GetParameters().At(0).GetValue()-1.792786e+00) > 1e-4 {
		t.Error("Hmm test failed!")
	}
	if math.Abs(hmm.Edist[0].GetParameters().At(1).GetValue()-6.371870e+00) > 1e-4 {
		t.Error("Hmm test failed!")
	}
	if math.Abs(hmm.Edist[1].GetParameters().At(0).GetValue()-4.855799e+00) > 1e-4 {
		t.Error("Hmm test failed!")
	}
	if math.Abs(hmm.Edist[1].GetParameters().At(1).GetValue()-1.299225e+00) > 1e-4 {
		t.Error("Hmm test failed!")
	}
}

func TestHmm6(t *testing.T) {
	var err error

	pi := NewVector(RealType, []float64{0.6, 0.4})

	var tr Matrix = NewMatrix(RealType, 2, 2,
		[]float64{0.7, 0.3, 0.4, 0.6})

	c1, _ := scalarDistribution.NewGammaDistribution(NewReal(0.5), NewReal(2.0))
	e1, _ := scalarEstimator.NewNumericEstimator(c1)
	c2, _ := scalarDistribution.NewGammaDistribution(NewReal(10.0), NewReal(2.0))
	e2, _ := scalarEstimator.NewNumericEstimator(c2)
	c3, _ := scalarDistribution.NewGammaDistribution(NewReal(1.5), NewReal(3.0))
	e3, _ := scalarEstimator.NewNumericEstimator(c3)
	c4, _ := scalarDistribution.NewGammaDistribution(NewReal(10.0), NewReal(3.0))
	e4, _ := scalarEstimator.NewNumericEstimator(c4)

	e1.Epsilon = 1e-6
	e2.Epsilon = 1e-6
	e3.Epsilon = 1e-6
	e4.Epsilon = 1e-6

	f1, _ := scalarEstimator.NewMixtureEstimator([]float64{0.5, 0.5}, []ScalarEstimator{e1, e2}, 0, 0)
	f2, _ := scalarEstimator.NewMixtureEstimator([]float64{0.3, 0.7}, []ScalarEstimator{e3, e4}, 0, 0)

	x := []ConstVector{
		NewVector(RealType, []float64{
			// r1 <- rgamma(100,  1,  2)
			// r2 <- rgamma(100, 10, 10)
			0.287905493, 1.128806993, 0.340662837, 0.762040120, 0.895168827, 1.121907180,
			0.785289185, 2.098988941, 0.218481764, 0.582749132, 0.184671591, 0.627610253,
			0.259363985, 0.482734782, 0.462957319, 1.021673942, 0.375795282, 0.946716191,
			0.058158397, 0.908592378, 0.257020555, 0.886418130, 0.279082511, 1.395380476,
			0.037464659, 1.354805030, 1.199380758, 0.973696784, 0.695123365, 0.651326152,
			0.239893746, 1.285828818, 0.448714197, 0.809773405, 0.053391201, 0.981037433,
			0.903226594, 1.589674918, 1.131871586, 1.417193465, 0.990755837, 0.938683859,
			0.034755610, 0.907196737, 0.372530444, 0.888760737, 0.652716929, 0.888622486,
			0.950738082, 0.965109828, 1.739767098, 1.093978098, 0.052025117, 0.672560695,
			0.120876028, 0.969396736, 0.469652811, 1.280768236, 0.801381892, 0.597040134,
			0.036999972, 0.811190524, 0.474145004, 0.916866738, 0.101415053, 1.207149647,
			0.158691869, 1.275789002, 0.551370133, 1.069041993, 1.006903230, 0.522681001,
			0.094642374, 0.739430911, 0.229197505, 1.091046229, 0.139118954, 1.412196125,
			1.291185942, 0.505529795, 0.010110837, 1.281485807, 3.667816424, 0.779621699,
			1.468537039, 0.579454329, 0.214613379, 1.060167054, 0.301199465, 1.293311210,
			0.056242467, 0.684725648, 1.535987736, 0.635313778, 0.557904019, 0.802140187,
			1.080885951, 0.827012647, 0.902496055, 0.822993393, 0.185910641, 0.820423703,
			0.478639505, 0.556863371, 0.207642444, 1.986453636, 0.876667112, 0.661249930,
			0.768101497, 0.675378800, 0.018283000, 0.899586080, 0.024228611, 1.552047742,
			0.655642314, 1.699937674, 0.164680318, 1.216198327, 0.786303696, 0.980736511,
			0.086853672, 0.645101978, 0.148284252, 1.235573865, 0.283720451, 1.104134379,
			0.877014127, 0.763261676, 0.063996700, 0.587321575, 0.212431929, 1.988579093,
			0.299666227, 1.367478835, 0.175930047, 0.603735314, 0.102472775, 1.147057810,
			0.620167930, 1.071669783, 0.136008877, 1.459272540, 0.845151922, 0.574659889,
			0.064055561, 1.250522442, 0.855332119, 1.429870651, 0.027647477, 0.795645148,
			0.473198492, 1.299872669, 0.174953881, 0.911043433, 0.457196311, 0.505988172,
			0.639990290, 1.339049451, 0.358640832, 1.541877285, 0.183328885, 1.200039050,
			0.344588565, 0.791695478, 0.238703472, 1.118366231, 0.176147228, 1.321904122,
			0.004603618, 0.889694318, 0.153528607, 0.958559361, 0.159836308, 0.744773466,
			0.710984669, 1.094417510, 2.136895825, 1.051879556, 0.669342828, 1.208982933,
			0.226796591, 1.413601008, 1.117730061, 1.432472561, 0.108333914, 0.615915638,
			0.049344939, 0.920447769, 1.242411739, 0.608369780, 1.075726981, 0.682748764,
			1.130705661, 0.793907131, 0.071857918, 0.952569108, 0.419231006, 0.867176265,
			0.280672333, 0.917844861,
			// r3 <- rgamma(100, 100, 2)
			// r4 <- rgamma(100, 140, 4)
			54.01832, 38.51624, 52.55921, 33.15652, 47.35385, 35.75663, 40.56186, 33.53133,
			53.65520, 36.40228, 45.76913, 36.03882, 44.77368, 37.98400, 55.36586, 36.39089,
			60.22218, 34.91533, 54.06612, 38.08065, 57.38789, 30.60221, 50.09213, 34.37447,
			58.15436, 31.59375, 53.43363, 36.86049, 54.76541, 32.36120, 49.10280, 36.30680,
			45.67872, 34.87615, 43.69334, 35.45732, 57.39876, 33.97928, 57.21684, 35.74039,
			51.53844, 32.76678, 47.17697, 32.51151, 52.78435, 36.18651, 55.20670, 35.95045,
			57.64610, 32.68200, 52.01257, 36.56077, 45.00477, 35.10799, 56.92544, 34.19623,
			40.81040, 35.04998, 42.46825, 32.96427, 49.25694, 36.66672, 48.79356, 36.93031,
			46.73011, 36.82630, 52.07839, 38.25338, 49.99606, 37.44097, 57.04304, 33.32736,
			52.59640, 38.68918, 49.44449, 35.61437, 52.75532, 38.66412, 43.20029, 37.09850,
			50.46193, 38.91167, 47.35258, 39.06894, 47.09457, 38.92572, 49.79819, 38.61649,
			50.90998, 32.74014, 48.72774, 41.64155, 63.56865, 33.62719, 48.14005, 32.84118,
			47.82962, 32.43160, 48.63155, 33.21634, 50.25004, 31.89347, 37.83782, 30.77140,
			52.10081, 31.42627, 55.02078, 39.30558, 51.84625, 38.13764, 49.07423, 36.34342,
			50.78435, 37.92456, 56.74490, 31.19618, 47.17544, 35.63038, 50.09087, 35.75053,
			53.67033, 33.28257, 54.55972, 31.68791, 55.78388, 37.77054, 41.65875, 29.87213,
			46.39157, 34.92732, 45.57285, 34.42843, 44.76154, 41.97071, 47.65849, 34.59747,
			52.54299, 32.26125, 52.05569, 35.67482, 42.72924, 38.82131, 54.98362, 34.06179,
			41.26299, 31.67788, 54.67701, 37.39823, 46.64685, 32.82941, 51.13384, 40.82067,
			55.19166, 33.64918, 52.88355, 32.71007, 54.78949, 34.49200, 52.01185, 35.93166,
			44.83625, 32.15400, 49.93046, 36.10971, 51.52567, 36.95301, 56.38854, 32.57861,
			47.30639, 34.35405, 40.81722, 39.35588, 56.86898, 30.20007, 43.57844, 31.92096,
			49.18540, 34.76059, 47.78013, 33.70723, 51.05670, 37.50723, 49.64923, 34.38490,
			52.53555, 36.78197, 51.54151, 39.34633, 42.58552, 36.37609, 52.56146, 37.34477,
			57.82587, 32.90822, 45.33739, 29.18969, 47.45135, 29.04017, 43.86988, 31.33080})}

	r := NewVector(RealType, []float64{
		-9.628283e+04, 0.000000e+00, 0.000000e+00, -2.811200e+03, -5.298317e+00, -5.012542e-03, // Hmm
		-2.851585e+02, 0.000000e+00, 4.262817e+01, 4.718743e+01, 2.513352e+01, 5.887617e-01, // Mixture component 1
		-6.486053e-01, -7.397659e-01, 9.793777e-01, 1.813242e+00, 1.139991e+01, 1.144401e+01}) // Mixture component 2
	r.Slice(0, 6).Map(func(x Scalar) { x.Exp(x) })
	r.Slice(12, 14).Map(func(x Scalar) { x.Exp(x) })

	estimator, err := NewHmmEstimator(pi, tr, nil, nil, nil, []ScalarEstimator{f1, f2}, 1e-8, -1)
	if err != nil {
		t.Error(err)
		return
	}
	if err := estimator.EstimateOnData(x, nil, ThreadPool{}); err != nil {
		t.Error(err)
		return
	}
	pt, _ := estimator.GetEstimate()
	p := pt.GetParameters()
	p.Slice(0, 6).Map(func(x Scalar) { x.Exp(x) })
	p.Slice(12, 14).Map(func(x Scalar) { x.Exp(x) })

	if Vnorm(VsubV(r, p)).GetValue() > 1e-4 {
		t.Error("test failed")
	}
}
