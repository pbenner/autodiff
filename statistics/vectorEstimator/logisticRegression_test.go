/* Copyright (C) 2019 Philipp Benner
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

import "fmt"
import "testing"

import . "github.com/pbenner/autodiff"
import "github.com/pbenner/autodiff/algorithm/rprop"
import "github.com/pbenner/autodiff/statistics/vectorDistribution"
import . "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func hook(x ConstVector, step, lambda ConstScalar, i int) bool {
	fmt.Printf("x: %v\n", x)
	fmt.Printf("s: %v\n", step)
	fmt.Printf("i: %d\n", i)
	fmt.Println()
	return false
}

func rprop_hook(gradient []float64, step, lambda []float64, x Vector, value Scalar) bool {
	fmt.Printf("y: %v\n", value)
	fmt.Printf("x: %v\n", x)
	fmt.Printf("g: %v\n", gradient)
	fmt.Println()
	return false
}

/* -------------------------------------------------------------------------- */

func eval_l1_solution(x []ConstVector, theta ConstVector, C float64) Scalar {
	v := AsDenseRealVector(theta)
	v.Variables(1)
	t := NewReal(0.0)
	s := NewReal(0.0)
	l := ConstReal(1.0 / C)
	if r, err := vectorDistribution.NewLogisticRegression(v); err != nil {
		panic(err)
	} else {
		for i, _ := range x {
			if err := r.ClassLogPdf(t, x[i].ConstSlice(0, x[i].Dim()-1), x[i].ValueAt(x[i].Dim()-1) == 1.0); err != nil {
				panic(err)
			}
			s.Add(s, t)
		}
		s.Neg(s)
		for i := 1; i < v.Dim(); i++ {
			t.Abs(v.At(i))
			t.Mul(t, l)
			s.Add(s, t)
		}
	}
	return s
}

func eval_l2_solution(x []ConstVector, theta ConstVector, C float64) Scalar {
	n := theta.Dim()
	v := AsDenseRealVector(theta)
	v.Variables(1)
	t := NewReal(0.0)
	s := NewReal(0.0)
	l := ConstReal(1.0 / C)
	if r, err := vectorDistribution.NewLogisticRegression(v); err != nil {
		panic(err)
	} else {
		for i, _ := range x {
			if err := r.ClassLogPdf(t, x[i].ConstSlice(0, x[i].Dim()-1), x[i].ValueAt(x[i].Dim()-1) == 1.0); err != nil {
				panic(err)
			}
			s.Add(s, t)
		}
		s.Neg(s)
		t.Vnorm(v[1:n])
		t.Mul(t, t)
		t.Mul(t, l)
		t.Mul(t, ConstReal(0.5))
		s.Add(s, t)
	}
	return s
}

/* -------------------------------------------------------------------------- */

func TestLogistic1(test *testing.T) {

	// data
	cellSize := []float64{
		1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
	cellShape := []float64{
		1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
	class := []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
	// x
	x := make([]ConstVector, len(cellSize))
	for i := 0; i < len(cellSize); i++ {
		x[i] = NewDenseBareRealVector([]float64{1.0, cellSize[i] - 1.0, cellShape[i] - 1.0, class[i]})
	}

	estimator, err := NewLogisticRegression(3, false)
	if err != nil {
		test.Error(err)
		return
	}
	//estimator.Hook = hook

	err = estimator.EstimateOnData(x, nil, ThreadPool{})
	if err != nil {
		test.Error(err)
		return
	}
	// result and target
	r := estimator.GetParameters()
	z := DenseConstRealVector([]float64{-2.858321e+00, 1.840900e-01, 5.067086e-01})
	t := NullReal()

	if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
}

func TestLogistic2(test *testing.T) {

	// data
	cellSize := []float64{
		1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
	cellShape := []float64{
		1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
	class := []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
	// x
	x := make([]ConstVector, len(cellSize))
	for i := 0; i < len(cellSize); i++ {
		x[i] = NewSparseBareRealVector([]int{0, 1, 2, 3}, []float64{1.0, cellSize[i] - 1.0, cellShape[i] - 1.0, class[i]}, 4)
	}

	estimator, err := NewLogisticRegression(3, true)
	if err != nil {
		test.Error(err)
		return
	}
	estimator.Epsilon = 1e-7
	//estimator.Hook = hook

	err = estimator.EstimateOnData(x, nil, ThreadPool{})
	if err != nil {
		test.Error(err)
		return
	}
	// result and target
	r := estimator.GetParameters()
	z := DenseConstRealVector([]float64{-2.858321e+00, 1.840900e-01, 5.067086e-01})
	t := NullReal()

	if t.Vnorm(r.VsubV(r, z)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
}

func TestLogistic3(test *testing.T) {

	C := 1.0

	// data
	cellSize := []float64{
		1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
	cellShape := []float64{
		1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
	class := []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
	// x
	x := make([]ConstVector, len(cellSize))
	for i := 0; i < len(cellSize); i++ {
		x[i] = NewDenseBareRealVector([]float64{1.0, cellSize[i] - 1.0, cellShape[i] - 1.0, class[i]})
	}

	estimator, err := NewLogisticRegression(3, false)
	if err != nil {
		test.Error(err)
		return
	}
	//estimator.Hook  = hook
	estimator.TiReg = 1.0 / C

	err = estimator.EstimateOnData(x, nil, ThreadPool{})
	if err != nil {
		test.Error(err)
		return
	}
	// result and target
	r_saga := estimator.GetParameters()

	objective := func(r Vector) (Scalar, error) {
		return eval_l2_solution(x, r, C), nil
	}
	r_rprop, err := rprop.Run(objective, NewDenseRealVector([]float64{1, 1, 1}), 0.01, []float64{2, 0.1}, rprop.Epsilon{1e-12})
	if err != nil {
		panic(err)
	}
	r_sklearn := DenseConstRealVector([]float64{-2.77092384, 0.25099466, 0.40914467})

	t := NullReal()
	s := NullDenseBareRealVector(r_saga.Dim())
	if t.Vnorm(s.VsubV(r_saga, r_rprop)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
	if t.Vnorm(s.VsubV(r_saga, r_sklearn)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
}

func TestLogistic4(test *testing.T) {

	C := 1.0

	// data
	cellSize := []float64{
		1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
	cellShape := []float64{
		1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
	class := []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
	// x
	x := make([]ConstVector, len(cellSize))
	for i := 0; i < len(cellSize); i++ {
		x[i] = NewDenseBareRealVector([]float64{1.0, cellSize[i] - 1.0, cellShape[i] - 1.0, class[i]})
	}

	estimator, err := NewLogisticRegression(3, false)
	if err != nil {
		test.Error(err)
		return
	}
	//estimator.Hook  = hook
	estimator.L1Reg = 1.0 / C

	err = estimator.EstimateOnData(x, nil, ThreadPool{})
	if err != nil {
		test.Error(err)
		return
	}
	// result and target
	r_saga := estimator.GetParameters()

	objective := func(r Vector) (Scalar, error) {
		return eval_l1_solution(x, r, C), nil
	}
	r_rprop, err := rprop.Run(objective, NewDenseRealVector([]float64{1, 1, 1}), 0.01, []float64{2, 0.1}, rprop.Epsilon{1e-12})
	if err != nil {
		panic(err)
	}
	r_sklearn := DenseConstRealVector([]float64{-2.63837871, 0.16460826, 0.44788412})

	t := NullReal()
	s := NullDenseBareRealVector(r_saga.Dim())
	if t.Vnorm(s.VsubV(r_saga, r_rprop)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
	if t.Vnorm(s.VsubV(r_saga, r_sklearn)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
}

func TestLogistic5(test *testing.T) {

	C := 2.5

	// data
	cellSize := []float64{
		1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
	cellShape := []float64{
		1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
	class := []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
	// x
	x := make([]ConstVector, len(cellSize))
	for i := 0; i < len(cellSize); i++ {
		x[i] = NewDenseBareRealVector([]float64{1.0, cellSize[i] - 1.0, cellShape[i] - 1.0, class[i]})
	}

	estimator, err := NewLogisticRegression(3, false)
	if err != nil {
		test.Error(err)
		return
	}
	//estimator.Hook  = hook
	estimator.TiReg = 1.0 / C

	err = estimator.EstimateOnData(x, nil, ThreadPool{})
	if err != nil {
		test.Error(err)
		return
	}
	// result and target
	r_saga := estimator.GetParameters()

	objective := func(r Vector) (Scalar, error) {
		return eval_l2_solution(x, r, C), nil
	}
	r_rprop, err := rprop.Run(objective, NewDenseRealVector([]float64{1, 1, 1}), 0.01, []float64{2, 0.1}, rprop.Epsilon{1e-12})
	if err != nil {
		panic(err)
	}
	r_sklearn := DenseConstRealVector([]float64{-2.81978662, 0.22409962, 0.45325202})

	t := NullReal()
	s := NullDenseBareRealVector(r_saga.Dim())
	if t.Vnorm(s.VsubV(r_saga, r_rprop)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
	if t.Vnorm(s.VsubV(r_saga, r_sklearn)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
}

func TestLogistic6(test *testing.T) {

	C := 2.5

	// data
	cellSize := []float64{
		1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
	cellShape := []float64{
		1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
	class := []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
	// x
	x := make([]ConstVector, len(cellSize))
	for i := 0; i < len(cellSize); i++ {
		x[i] = NewDenseBareRealVector([]float64{1.0, cellSize[i] - 1.0, cellShape[i] - 1.0, class[i]})
	}

	estimator, err := NewLogisticRegression(3, false)
	if err != nil {
		test.Error(err)
		return
	}
	//estimator.Hook  = hook
	estimator.L1Reg = 1.0 / C

	err = estimator.EstimateOnData(x, nil, ThreadPool{})
	if err != nil {
		test.Error(err)
		return
	}
	// result and target
	r_saga := estimator.GetParameters()

	objective := func(r Vector) (Scalar, error) {
		return eval_l1_solution(x, r, C), nil
	}
	r_rprop, err := rprop.Run(objective, NewDenseRealVector([]float64{1, 1, 1}), 0.01, []float64{2, 0.1}, rprop.Epsilon{1e-12})
	if err != nil {
		panic(err)
	}
	r_sklearn := DenseConstRealVector([]float64{-2.76467776, 0.17584927, 0.48174453})

	t := NullReal()
	s := NullDenseBareRealVector(r_saga.Dim())
	if t.Vnorm(s.VsubV(r_saga, r_rprop)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
	if t.Vnorm(s.VsubV(r_saga, r_sklearn)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
}

func TestLogistic7(test *testing.T) {

	C := 1.0

	// data
	cellSize := []float64{
		1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
	cellShape := []float64{
		1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
	class := []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
	// x
	x := make([]ConstVector, len(cellSize))
	for i := 0; i < len(cellSize); i++ {
		x[i] = NewDenseBareRealVector([]float64{1.0, cellSize[i] - 1.0, cellShape[i] - 1.0, class[i]})
	}

	estimator, err := NewLogisticRegression(3, true)
	if err != nil {
		test.Error(err)
		return
	}
	//estimator.Hook    = hook
	estimator.Epsilon = 1e-8
	estimator.L1Reg = 1.0 / C

	err = estimator.EstimateOnData(x, nil, ThreadPool{})
	if err != nil {
		test.Error(err)
		return
	}
	// result and target
	r_saga := estimator.GetParameters()

	objective := func(r Vector) (Scalar, error) {
		return eval_l1_solution(x, r, C), nil
	}
	r_rprop, err := rprop.Run(objective, NewDenseRealVector([]float64{1, 1, 1}), 0.01, []float64{2, 0.1}, rprop.Epsilon{1e-12})
	if err != nil {
		panic(err)
	}
	r_sklearn := DenseConstRealVector([]float64{-2.63837871, 0.16460826, 0.44788412})

	t := NullReal()
	s := NullDenseBareRealVector(r_saga.Dim())
	if t.Vnorm(s.VsubV(r_saga, r_rprop)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
	if t.Vnorm(s.VsubV(r_saga, r_sklearn)); t.GetValue() > 1e-4 {
		test.Error("test failed")
	}
}

func TestLogistic8(test *testing.T) {

	C := 0.2

	// data
	cellSize := []float64{
		1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
	cellShape := []float64{
		1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
	class := []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
	// x
	x := make([]ConstVector, len(cellSize))
	for i := 0; i < len(cellSize); i++ {
		x[i] = NewDenseBareRealVector([]float64{1.0, cellSize[i] - 1.0, cellShape[i] - 1.0, class[i]})
	}

	trace1 := []ConstVector{}
	// result from sklearn
	trace2 := NewDenseBareRealMatrix(10, 3, []float64{
		-7.656644e-02, 8.506265e-02, 9.060131e-02,
		-1.423653e-01, 5.056287e-02, 5.386971e-02,
		-1.951594e-01, 5.957177e-02, 6.109813e-02,
		-2.650173e-01, 5.160914e-02, 6.153448e-02,
		-3.212615e-01, 4.132712e-02, 5.023932e-02,
		-3.725816e-01, 4.718922e-02, 5.600616e-02,
		-4.222260e-01, 5.004562e-02, 5.936789e-02,
		-4.686855e-01, 5.620630e-02, 6.600405e-02,
		-5.135289e-01, 5.841420e-02, 6.871803e-02,
		-5.571380e-01, 6.193702e-02, 7.279218e-02})
	hook_record := func(x ConstVector, step, lambda ConstScalar, i int) bool {
		// clone vector!
		trace1 = append(trace1, AsDenseBareRealVector(x))
		return false
	}
	estimator, err := NewLogisticRegression(3, true)
	if err != nil {
		test.Error(err)
		return
	}
	estimator.Hook = hook_record
	estimator.L1Reg = 1.0 / C
	estimator.MaxIterations = 10
	estimator.Seed = -1

	err = estimator.EstimateOnData(x, nil, ThreadPool{})
	if err != nil {
		test.Error(err)
		return
	}

	if nr, _ := trace2.Dims(); len(trace1) == 0 || len(trace1) != nr {
		test.Error("test failed")
	} else {
		t1 := NullReal()
		t2 := NullDenseBareRealVector(3)
		for i := 0; i < len(trace1); i++ {
			if t1.Vnorm(t2.VsubV(trace1[i], trace2.Row(i))); t1.GetValue() > 1e-4 {
				test.Error("test failed")
			}
		}
	}
}

func TestLogistic9(test *testing.T) {

	C := 0.2

	// data
	cellSize := []float64{
		1, 4, 1, 8, 1, 10, 1, 1, 1, 2, 1, 1, 3, 1, 7, 4, 1, 1, 7, 1}
	cellShape := []float64{
		1, 4, 1, 8, 1, 10, 1, 2, 1, 1, 1, 1, 3, 1, 5, 6, 1, 1, 7, 1}
	class := []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}
	// x
	x := make([]ConstVector, len(cellSize))
	for i := 0; i < len(cellSize); i++ {
		x[i] = NewDenseBareRealVector([]float64{1.0, cellSize[i] - 1.0, cellShape[i] - 1.0, class[i]})
	}

	trace1 := []ConstVector{}
	// result from sklearn
	trace2 := NewDenseBareRealMatrix(100, 3, []float64{
		-1.098232e-01, 1.085653e-01, 8.986232e-02,
		-1.867656e-01, 9.044017e-02, 8.480379e-02,
		-2.662775e-01, 5.905653e-02, 5.416249e-02,
		-3.290223e-01, 4.317760e-02, 3.718732e-02,
		-3.809268e-01, 5.501890e-02, 4.870041e-02,
		-4.299158e-01, 5.774855e-02, 5.348177e-02,
		-4.765570e-01, 6.132310e-02, 5.787355e-02,
		-5.213309e-01, 6.460371e-02, 6.211760e-02,
		-5.645407e-01, 6.885071e-02, 6.733383e-02,
		-6.058630e-01, 7.077392e-02, 7.017612e-02,
		-6.456605e-01, 7.434266e-02, 7.480967e-02,
		-6.836872e-01, 7.896873e-02, 8.027105e-02,
		-7.212598e-01, 8.040837e-02, 8.248646e-02,
		-7.571243e-01, 8.301525e-02, 8.580488e-02,
		-7.913767e-01, 8.593513e-02, 8.974701e-02,
		-8.252745e-01, 8.827525e-02, 9.326077e-02,
		-8.573683e-01, 8.983893e-02, 9.573680e-02,
		-8.880659e-01, 9.276461e-02, 9.977766e-02,
		-9.181512e-01, 9.445395e-02, 1.026233e-01,
		-9.471123e-01, 9.608982e-02, 1.054181e-01,
		-9.750991e-01, 9.811849e-02, 1.085492e-01,
		-1.002323e+00, 1.002029e-01, 1.117745e-01,
		-1.028566e+00, 1.018700e-01, 1.144960e-01,
		-1.053738e+00, 1.043926e-01, 1.182275e-01,
		-1.078502e+00, 1.054667e-01, 1.205101e-01,
		-1.102313e+00, 1.067615e-01, 1.229508e-01,
		-1.125402e+00, 1.083352e-01, 1.259723e-01,
		-1.147596e+00, 1.092178e-01, 1.278391e-01,
		-1.169151e+00, 1.108947e-01, 1.308590e-01,
		-1.190202e+00, 1.118719e-01, 1.329221e-01,
		-1.210747e+00, 1.134132e-01, 1.356085e-01,
		-1.230393e+00, 1.145665e-01, 1.379175e-01,
		-1.249693e+00, 1.159922e-01, 1.404669e-01,
		-1.268396e+00, 1.162867e-01, 1.422975e-01,
		-1.286828e+00, 1.173386e-01, 1.446930e-01,
		-1.304533e+00, 1.182607e-01, 1.470002e-01,
		-1.321889e+00, 1.190542e-01, 1.492754e-01,
		-1.338798e+00, 1.199259e-01, 1.516587e-01,
		-1.355296e+00, 1.207225e-01, 1.540048e-01,
		-1.371123e+00, 1.211103e-01, 1.553402e-01,
		-1.386401e+00, 1.223169e-01, 1.577400e-01,
		-1.401309e+00, 1.228188e-01, 1.595467e-01,
		-1.415988e+00, 1.236848e-01, 1.615281e-01,
		-1.429877e+00, 1.244023e-01, 1.634591e-01,
		-1.443735e+00, 1.246094e-01, 1.649447e-01,
		-1.457099e+00, 1.253929e-01, 1.668075e-01,
		-1.469980e+00, 1.259035e-01, 1.685399e-01,
		-1.482591e+00, 1.263438e-01, 1.701734e-01,
		-1.494831e+00, 1.270416e-01, 1.720737e-01,
		-1.506764e+00, 1.276969e-01, 1.738035e-01,
		-1.518337e+00, 1.278470e-01, 1.750865e-01,
		-1.529715e+00, 1.282026e-01, 1.766388e-01,
		-1.540869e+00, 1.285327e-01, 1.781344e-01,
		-1.551512e+00, 1.290839e-01, 1.798845e-01,
		-1.561969e+00, 1.293261e-01, 1.812857e-01,
		-1.572183e+00, 1.297862e-01, 1.828426e-01,
		-1.582179e+00, 1.300167e-01, 1.843046e-01,
		-1.591921e+00, 1.303664e-01, 1.858246e-01,
		-1.601454e+00, 1.309341e-01, 1.874885e-01,
		-1.610778e+00, 1.309667e-01, 1.887016e-01,
		-1.619836e+00, 1.313105e-01, 1.902177e-01,
		-1.628719e+00, 1.314383e-01, 1.914267e-01,
		-1.637391e+00, 1.314157e-01, 1.924629e-01,
		-1.645714e+00, 1.316463e-01, 1.937677e-01,
		-1.653969e+00, 1.318340e-01, 1.950398e-01,
		-1.662087e+00, 1.320923e-01, 1.963967e-01,
		-1.670002e+00, 1.322213e-01, 1.976384e-01,
		-1.677742e+00, 1.323146e-01, 1.988257e-01,
		-1.685278e+00, 1.325512e-01, 2.001104e-01,
		-1.692703e+00, 1.325581e-01, 2.011561e-01,
		-1.699916e+00, 1.326817e-01, 2.023340e-01,
		-1.706878e+00, 1.328851e-01, 2.036264e-01,
		-1.713773e+00, 1.328324e-01, 2.046067e-01,
		-1.720505e+00, 1.329129e-01, 2.057589e-01,
		-1.727058e+00, 1.329642e-01, 2.068103e-01,
		-1.733454e+00, 1.330121e-01, 2.078727e-01,
		-1.739671e+00, 1.331695e-01, 2.090712e-01,
		-1.745812e+00, 1.331165e-01, 2.100183e-01,
		-1.751814e+00, 1.330908e-01, 2.110143e-01,
		-1.757600e+00, 1.330518e-01, 2.119906e-01,
		-1.763285e+00, 1.330544e-01, 2.129676e-01,
		-1.768869e+00, 1.331140e-01, 2.140344e-01,
		-1.774313e+00, 1.332028e-01, 2.150818e-01,
		-1.779692e+00, 1.331263e-01, 2.159949e-01,
		-1.784922e+00, 1.330906e-01, 2.169385e-01,
		-1.790043e+00, 1.330080e-01, 2.178239e-01,
		-1.795050e+00, 1.328658e-01, 2.187011e-01,
		-1.799932e+00, 1.330625e-01, 2.198327e-01,
		-1.804714e+00, 1.330570e-01, 2.207910e-01,
		-1.809419e+00, 1.328505e-01, 2.215040e-01,
		-1.813995e+00, 1.327987e-01, 2.223458e-01,
		-1.818481e+00, 1.327922e-01, 2.232497e-01,
		-1.822890e+00, 1.326907e-01, 2.240540e-01,
		-1.827203e+00, 1.326252e-01, 2.248878e-01,
		-1.831455e+00, 1.325712e-01, 2.256931e-01,
		-1.835526e+00, 1.325983e-01, 2.266067e-01,
		-1.839572e+00, 1.326031e-01, 2.274872e-01,
		-1.843573e+00, 1.323689e-01, 2.281278e-01,
		-1.847517e+00, 1.323090e-01, 2.289377e-01,
		-1.851317e+00, 1.322393e-01, 2.297577e-01})
	hook_record := func(x ConstVector, step, lambda ConstScalar, i int) bool {
		// clone vector!
		trace1 = append(trace1, AsDenseBareRealVector(x))
		return false
	}
	estimator, err := NewLogisticRegression(3, true)
	if err != nil {
		test.Error(err)
		return
	}
	estimator.Hook = hook_record
	estimator.L1Reg = 1.0 / C
	estimator.MaxIterations = 100
	estimator.Seed = 1

	err = estimator.EstimateOnData(x, nil, ThreadPool{})
	if err != nil {
		test.Error(err)
		return
	}

	if nr, _ := trace2.Dims(); len(trace1) == 0 || len(trace1) != nr {
		test.Error("test failed")
	} else {
		t1 := NullReal()
		t2 := NullDenseBareRealVector(3)
		for i := 0; i < len(trace1); i++ {
			if t1.Vnorm(t2.VsubV(trace1[i], trace2.Row(i))); t1.GetValue() > 1e-4 {
				test.Error("test failed")
			}
		}
	}
}
