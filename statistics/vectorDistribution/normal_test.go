/* Copyright (C) 2016 Philipp Benner
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

package vectorDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"
import "math"
import "testing"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/rprop"
import . "github.com/pbenner/autodiff/simple"

/* -------------------------------------------------------------------------- */

func TestNormalDistribution1(t *testing.T) {

	mu := NewVector(RealType, []float64{2, 3})
	sigma := NewMatrix(RealType, 2, 2, []float64{2, 1, 1, 2})

	normal, _ := NewNormalDistribution(mu, sigma)

	x := NewVector(RealType, []float64{1, 2})
	y := NewReal(0.0)

	Variables(1, x.At(0), x.At(1))

	normal.LogPdf(y, x)

	if math.Abs(y.GetValue() - -2.720517) > 1e-4 {
		t.Error("Normal LogPdf failed!")
	}
	if math.Abs(y.GetDerivative(0)-0.3333333) > 1e-4 {
		t.Error("Normal LogPdf failed!")
	}
	if math.Abs(y.GetDerivative(1)-0.3333333) > 1e-4 {
		t.Error("Normal LogPdf failed!")
	}
}

func TestNormalDistribution2(t *testing.T) {

	mu := NewVector(RealType, []float64{2, 3})
	sigma := NewMatrix(RealType, 2, 2, []float64{16, 8, 8, 10})

	normal, _ := NewNormalDistribution(mu, sigma)

	x := NewVector(RealType, []float64{1, 2})
	y := NewReal(0.0)

	normal.LogPdf(y, x)

	if math.Abs(y.GetValue() - -4.172134e+00) > 1e-4 {
		t.Error("Normal LogPdf failed!")
	}
}

func TestNormalDistribution3(t *testing.T) {

	mu := NewVector(RealType, []float64{3})
	sigma := NewMatrix(RealType, 1, 1, []float64{2})

	normal, _ := NewNormalDistribution(mu, sigma)

	x := NewVector(RealType, []float64{2.2})
	y := NewReal(0.0)

	Variables(1, x.At(0))

	normal.LogPdf(y, x)

	if math.Abs(y.GetDerivative(0)-0.4) > 1e-4 {
		t.Error("Normal LogPdf failed!")
	}
}

func TestNormalDistribution4(t *testing.T) {

	m := NewReal(3.0)
	s := NewReal(4.0)

	Variables(1, m, s)

	mu := NewVector(RealType, []float64{3})
	sigma := NewMatrix(RealType, 1, 1, []float64{4})

	mu.At(0).Set(m)
	sigma.At(0, 0).Set(s)

	normal, _ := NewNormalDistribution(mu, sigma)

	x := NewVector(RealType, []float64{0.32094})
	y := NewReal(0.0)

	normal.LogPdf(y, x)

	if math.Abs(y.GetDerivative(0) - -0.669765) > 1e-4 {
		t.Error("Normal LogPdf failed!")
	}
}

func TestNormalFit1(t *testing.T) {
	// define the observed data
	x := []float64{
		0.3209406, 3.6823788, 3.2565888, 4.4076078, 1.2086832, 4.4392755,
		5.8210895, 1.8352054, 1.4654018, 2.4416984, 3.0245664, 5.6995920,
		2.6090983, 0.8538849, 1.5743261, 3.5348501, 2.6010283, 6.2657223,
		2.6263666, -1.9196041, 1.6093995, 5.1631672, 2.7381549, 4.5491679,
		0.2554603, 4.9089255, 4.9768131, 0.7276424, 2.1797271, 3.2427158,
		4.9625528, 5.1098779, 0.4970120, 1.3110702, 2.7368350, -0.2808708,
		0.5916626, 4.2850868, 6.4179787, 1.6638995, 3.8909360, 5.1452523,
		4.8637462, 0.4321876, 3.8595342, 8.8371891, 1.6073322, 5.8445747,
		5.7926215, 1.0415648, 7.0445693, 2.0619239, 2.1078349, 4.8303393,
		8.3079950, 6.0157360, 2.0648332, 5.1945344, 4.6980468, 2.8762338,
		4.7953537, 3.0568233, 5.9238065, 3.9467582, 2.5002062, 1.5441064,
		4.9109106, 3.3505682, 0.1572325, 2.1132299, 4.7223003, 3.5987367,
		2.7204970, -1.1296890, -0.3059554, 1.1185159, 3.4126256, 0.7536641,
		2.7735972, 3.6713431, 2.7675115, 1.7174240, 1.6777091, 8.4550003,
		1.2257446, 3.3113684, 5.1801388, 2.2084823, 1.4010860, 3.7119648,
		6.1073776, 7.1421863, 1.0146286, 3.7467156, 1.4064666, 2.3686035,
		4.3714954, 4.0999415, 2.2389224, 6.1477621}
	y := NewReal(0.0)
	// define the (negative) likelihood function (here as a function of the
	// variables that we want to optimize)
	objective := func(variables Vector) (Scalar, error) {
		// create a new initial normal distribution
		mu := NullVector(RealType, 1)
		sigma := NullMatrix(RealType, 1, 1)
		// copy the variables
		mu.At(0).Set(variables.At(0))
		sigma.At(0, 0).Set(variables.At(1))
		normal, _ := NewNormalDistribution(mu, sigma)
		result := NewScalar(RealType, 0.0)
		for i := 0; i < len(x); i++ {
			normal.LogPdf(y, NewVector(RealType, []float64{x[i]}))
			result.Add(result, y)
		}
		return Neg(result), nil
	}
	// rprop hook
	// hook := func(gradient []float64, variables Vector, s Scalar) bool {
	//   fmt.Println("variables:", variables)
	//   fmt.Println("gradient :", gradient)
	//   fmt.Println()
	//   return false
	// }
	// initial value
	v0 := NewVector(RealType, []float64{3, 4})
	// run rprop
	vn, _ := rprop.Run(objective, v0, 0.01, []float64{1.1, 0.1},
		//rprop.Hook{hook},
		rprop.Epsilon{1e-8})
	// check result
	if math.Abs(vn.At(0).GetValue()-3.238471) > 1e-4 ||
		math.Abs(vn.At(1).GetValue()-4.502649) > 1e-4 {
		t.Error("TestNormalFit1 failed!")
	}
}

func TestNormalFit2(t *testing.T) {
	// define the observed data
	x := NewMatrix(RealType, 100, 2, []float64{
		3.1698655, 2.07178555,
		1.4084200, 1.61097261,
		3.6935883, 2.95324698,
		2.2919326, 1.14012430,
		1.2972671, 0.43916031,
		2.5230651, 3.79502405,
		3.5803436, 1.67128667,
		1.7295307, 3.26372325,
		1.5235178, 2.08278233,
		0.9786134, 3.42009007,
		3.5613272, 5.10083231,
		1.4008066, 3.83257972,
		3.3034712, 1.18671979,
		5.4409581, 4.68237566,
		2.4002125, 3.09949661,
		5.3836190, 4.78778656,
		2.4002790, 1.66773750,
		0.1938836, 2.67515090,
		3.4721495, 2.39474629,
		1.0211469, 0.81974699,
		0.4198884, 2.25027930,
		0.6893919, 2.33546425,
		2.9822146, 4.94537819,
		1.3226016, 3.08623798,
		2.9617872, 2.91463207,
		0.3288441, 4.15332306,
		0.2405514, 1.62578651,
		1.1664488, 3.31667557,
		3.3452468, 2.81717879,
		1.9278152, 4.95226542,
		2.0991593, 6.34173263,
		2.6150638, 1.52666727,
		3.5001138, 1.65327658,
		1.4185267, 3.38222650,
		0.9728086, 3.86157957,
		1.1667347, 3.38989024,
		0.8001825, 2.79099041,
		3.5125261, 4.52682050,
		3.3888859, 3.82274572,
		3.3431360, 0.91933698,
		1.9939642, 3.46636865,
		2.1479620, 2.90371727,
		4.0612380, 3.63089449,
		2.1560887, 4.48165155,
		-0.2227403, 1.86279491,
		2.2731606, 5.47372856,
		3.8294974, 4.55647030,
		0.5393777, 1.96944494,
		2.2541245, 4.47840445,
		3.3035029, 3.01101497,
		1.0444364, 3.71425252,
		0.8985134, 0.47129126,
		0.5768917, 3.45433295,
		3.3505623, 3.19383569,
		1.7941806, 0.58568292,
		2.3688202, 1.79503268,
		2.7075895, 2.04168764,
		0.9812430, 2.79967514,
		0.9781523, 2.79007858,
		0.3133812, 1.04015794,
		2.4263145, 5.84338273,
		2.8537862, 3.79907603,
		1.6989009, 3.31484858,
		2.9657786, 4.88140609,
		2.3193185, 5.27937210,
		2.7516725, 2.94841728,
		4.4359379, 3.98506185,
		-1.2383294, 2.74763748,
		5.1041797, 3.69262452,
		2.3407159, 4.28864928,
		3.4634269, 6.74578866,
		1.8434792, 1.85393037,
		1.7718602, 0.69053583,
		-0.1477456, 3.72394865,
		1.5358038, 2.36489408,
		5.6644519, 3.41225292,
		3.2802483, 2.48674238,
		0.8724533, 3.35356573,
		2.6236578, 4.04150537,
		3.5788670, 5.46670610,
		2.8478828, 2.39031817,
		1.0657428, 2.53317961,
		2.6819949, 5.18504545,
		-0.8041424, 1.53044134,
		1.6864477, 3.03297775,
		0.8889615, 0.65069059,
		3.5186300, 4.24690838,
		3.0634319, 4.44820881,
		1.5673136, 2.33310465,
		-0.6334448, 1.42356262,
		3.3284329, 3.33202160,
		2.3087001, 4.30831412,
		-0.2892974, 2.61238485,
		3.8245939, 6.19400699,
		2.0315863, 1.13861252,
		1.6181918, 5.19722869,
		0.5486213, 3.90771551,
		1.0723826, 2.14096739,
		-0.4872921, -0.07513201,
		0.6451411, 3.54314968})
	y := NewReal(0.0)
	// number of data points
	n, _ := x.Dims()
	// define the (negative) likelihood function (here as a function of the
	// variables that we want to optimize)
	objective := func(variables Vector) (Scalar, error) {
		// create a new initial normal distribution
		mu := NullVector(RealType, 2)
		sigma := NullMatrix(RealType, 2, 2)
		// copy the variables
		mu.At(0).Set(variables.At(0))
		mu.At(1).Set(variables.At(1))
		sigma.At(0, 0).Set(variables.At(2))
		sigma.At(0, 1).Set(variables.At(3))
		sigma.At(1, 0).Set(variables.At(3))
		sigma.At(1, 1).Set(variables.At(4))
		normal, _ := NewNormalDistribution(mu, sigma)
		result := NewScalar(RealType, 0.0)
		for i := 0; i < n; i++ {
			normal.LogPdf(y, x.Row(i))
			result.Add(result, y)
		}
		return Neg(result), nil
	}
	// rprop hook
	// hook := func(gradient []float64, variables Vector, s Scalar) bool {
	//   fmt.Println("gradient :", gradient)
	//   fmt.Println("variables:", variables)
	//   return false
	// }
	// initial value
	v0 := NewVector(RealType, []float64{1, 1, 2, 1, 2})
	// run rprop
	vn, _ := rprop.Run(objective, v0, 0.01, []float64{1.2, 0.2},
		//rprop.Hook{hook},
		rprop.Epsilon{1e-8})
	// check result
	if math.Abs(vn.At(0).GetValue()-2.069545e+00) > 1e-4 ||
		math.Abs(vn.At(1).GetValue()-3.100224e+00) > 1e-4 ||
		math.Abs(vn.At(2).GetValue()-1.969915e+00) > 1e-4 ||
		math.Abs(vn.At(3).GetValue()-7.598206e-01) > 1e-4 ||
		math.Abs(vn.At(4).GetValue()-2.089113e+00) > 1e-4 {
		t.Error("TestNormalFit2 failed!")
	}
}
