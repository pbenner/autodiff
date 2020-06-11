/* Copyright (C) 2015 Philipp Benner
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

package autodiff

/* -------------------------------------------------------------------------- */

//import "fmt"
import "encoding/json"
import "math"
import "io/ioutil"
import "os"
import "testing"

/* -------------------------------------------------------------------------- */

func TestReal(t *testing.T) {

	a := NewReal(1.0)

	if a.GetValue() != 1.0 {
		t.Error("a.GetValue() should be 1.0")
	}

	a.Add(a, NewReal(2.0))

	if a.GetValue() != 3.0 {
		t.Error("a.GetValue() should be 3.0")
	}
}

func TestDiff1(t *testing.T) {

	t1 := NullReal()

	f := func(x Scalar) Scalar {
		return t1.Add(t1.Mul(NewReal(2), t1.Pow(x, NewBareReal(3))), NewReal(4))
	}
	x := NewReal(9)

	Variables(2, x)

	y := f(x)

	if y.GetDerivative(0) != 486 {
		t.Error("Differentiation failed!")
	}
	if y.GetHessian(0, 0) != 108 {
		t.Error("Differentiation failed!")
	}
}

func TestDiff2(t *testing.T) {

	f := func(x Scalar) Scalar {
		y := x.CloneScalar()
		y.Pow(y, NewBareReal(3))
		y.Mul(y, NewReal(2))
		y.Add(y, NewReal(4))
		return y
	}
	x := NewReal(9)

	Variables(2, x)

	y := f(x)

	if y.GetDerivative(0) != 486 {
		t.Error("Differentiation failed!")
	}
	if y.GetHessian(0, 0) != 108 {
		t.Error("Differentiation failed!")
	}
}

func TestMul(t *testing.T) {

	a := NewReal(13.123)
	b := NewReal(4.321)

	Variables(2, a)

	a.Mul(a, a) // a^2
	a.Mul(a, a) // a^4
	a.Mul(a, b) // a^4 b

	if math.Abs(a.GetValue()-128149.4603376) > 1e-4 {
		t.Error("Multiplication failed!")
	}
	if math.Abs(a.GetDerivative(0)-39061.025783) > 1e-4 {
		t.Error("Differentiation failed!")
	}
	if math.Abs(a.GetHessian(0, 0)-8929.5951649) > 1e-4 {
		t.Error("Differentiation failed!")
	}
}

func TestPow1(t *testing.T) {
	x := NewReal(3.4)
	k := NewReal(4.1)

	Variables(2, x, k)

	r := NullReal()
	r.Pow(x, k)

	if math.Abs(r.GetDerivative(0)-182.124553) > 1e-4 ||
		(math.Abs(r.GetDerivative(1)-184.826947) > 1e-4) {
		t.Error("Pow failed!")
	}
	if math.Abs(r.GetHessian(0, 0)-166.054739) > 1e-4 ||
		(math.Abs(r.GetHessian(1, 1)-226.186676) > 1e-4) {
		t.Error("Pow failed!")
	}
}

func TestPow2(t *testing.T) {
	x := NewReal(-3.4)
	k := NewReal(4.0)

	Variables(2, x, k)

	r := NullReal()
	r.Pow(x, k)

	if math.Abs(r.GetDerivative(0) - -157.216) > 1e-4 ||
		(math.Abs(r.GetHessian(0, 0)-138.720) > 1e-4) {
		t.Error("Pow failed!")
	}
	if !math.IsNaN(r.GetDerivative(1)) ||
		(!math.IsNaN(r.GetDerivative(1))) {
		t.Error("Pow failed!")
	}
}

func TestTan(t *testing.T) {

	a := NewReal(4.321)
	Variables(1, a)

	s := NullReal()
	s.Tan(a)

	if math.Abs(s.GetDerivative(0)-6.87184) > 0.0001 {
		t.Error("Incorrect derivative for Tan()!", s.GetDerivative(0))
	}
}

func TestTanh1(t *testing.T) {

	a := NewReal(4.321)
	Variables(2, a)

	s := NullReal()
	s.Tanh(a)

	if math.Abs(s.GetDerivative(0)-0.00070588) > 0.0000001 {
		t.Error("Incorrect derivative for Tanh()!")
	}
	if math.Abs(s.GetHessian(0, 0) - -0.00141127) > 0.0000001 {
		t.Error("Incorrect derivative for Tanh()!")
	}
}

func TestTanh2(t *testing.T) {

	a := NewReal(4.321)
	Variables(2, a)

	a.Tanh(a)

	if math.Abs(a.GetDerivative(0)-0.00070588) > 0.0000001 {
		t.Error("Incorrect derivative for Tanh()!")
	}
	if math.Abs(a.GetHessian(0, 0) - -0.00141127) > 0.0000001 {
		t.Error("Incorrect derivative for Tanh()!")
	}
}

func TestErf(t *testing.T) {

	a := NewReal(0.23)
	Variables(2, a)

	s := NullReal()
	s.Erf(a)

	if math.Abs(s.GetDerivative(0)-1.07023926) > 1e-6 ||
		(math.Abs(s.GetHessian(0, 0) - -0.49231006) > 1e-6) {
		t.Error("Incorrect derivative for Erf()!")
	}
}

func TestErfc(t *testing.T) {

	a := NewReal(0.23)
	Variables(2, a)

	s := NullReal()
	s.Erfc(a)

	if math.Abs(s.GetDerivative(0) - -1.07023926) > 1e-6 ||
		(math.Abs(s.GetHessian(0, 0)-0.49231006) > 1e-6) {
		t.Error("Incorrect derivative for Erfc()!")
	}
}

func TestLogErfc1(t *testing.T) {

	a := NewReal(0.23)
	Variables(2, a)

	s := NullReal()
	s.LogErfc(a)

	if math.Abs(s.GetDerivative(0) - -1.436606354) > 1e-6 {
		t.Error("Incorrect derivative for LogErfc()!")
	}
	if math.Abs(s.GetHessian(0, 0) - -1.402998894) > 1e-6 {
		t.Error("Incorrect derivative for LogErfc()!")
	}
}

func TestLogErfc2(t *testing.T) {

	a := NewReal(0.23)
	Variables(2, a)

	a.LogErfc(a)

	if math.Abs(a.GetDerivative(0) - -1.436606354) > 1e-6 {
		t.Error("Incorrect derivative for LogErfc()!")
	}
	if math.Abs(a.GetHessian(0, 0) - -1.402998894) > 1e-6 {
		t.Error("Incorrect derivative for LogErfc()!")
	}
}

func TestGamma(t *testing.T) {

	a := NewReal(4.321)
	Variables(2, a)

	s := NullReal()
	s.Gamma(a)

	if math.Abs(s.GetDerivative(0)-12.2353264) > 1e-6 ||
		(math.Abs(s.GetHessian(0, 0)-18.8065398) > 1e-6) {
		t.Error("Incorrect derivative for Gamma()!")
	}
}

func TestGammaP(t *testing.T) {

	x := NewReal(4.321)
	Variables(2, x)

	s := NullReal()
	s.GammaP(9.125, x)

	if math.Abs(s.GetValue()-0.029234) > 1e-6 ||
		(math.Abs(s.GetDerivative(0)-0.036763) > 1e-6) ||
		(math.Abs(s.GetHessian(0, 0)-0.032364) > 1e-6) {
		t.Error("Incorrect derivative for Gamma()!")
	}
}

func TestLogBessel(t *testing.T) {

	v := NewReal(10.0)
	x := NewReal(20.0)
	r := NewReal(0.0)

	Variables(2, x)

	r.LogBesselI(v.GetValue(), x)

	if math.Abs(r.GetValue()-15.0797) > 1e-4 {
		t.Error("Differentiation failed!")
	}
	if math.Abs(r.GetDerivative(0)-1.09804) > 1e-4 {
		t.Error("Differentiation failed!")
	}
	if math.Abs(r.GetHessian(0, 0) - -0.0106002) > 1e-4 {
		t.Error("Differentiation failed!")
	}

}

func TestHessian(t *testing.T) {
	x := NewReal(1.5)
	y := NewReal(2.5)
	k := NewReal(3.0)

	Variables(2, x, y)

	t1 := NullReal()
	t2 := NullReal()
	// y = x^3 + y^3 - 3xy
	z := NullReal()
	z.Sub(t1.Add(t1.Pow(x, k), t2.Pow(y, k)), t2.Mul(NewReal(3.0), t2.Mul(x, y)))

	if math.Abs(z.GetHessian(0, 0)-9) > 1e-6 ||
		(math.Abs(z.GetHessian(0, 1) - -3) > 1e-6) ||
		(math.Abs(z.GetHessian(1, 0) - -3) > 1e-6) ||
		(math.Abs(z.GetHessian(1, 1)-15) > 1e-6) {
		t.Error("Hessian test failed!")
	}
}

func TestRealJson(t *testing.T) {

	writeJson := func(filename string, obj interface{}) error {
		if f, err := os.Create(filename); err != nil {
			return err
		} else {
			b, err := json.MarshalIndent(obj, "", "  ")
			if err != nil {
				return err
			}
			if _, err := f.Write(b); err != nil {
				return err
			}
		}
		return nil
	}
	readJson := func(filename string, obj interface{}) error {
		if f, err := os.Open(filename); err != nil {
			return err
		} else {
			buffer, err := ioutil.ReadAll(f)
			if err != nil {
				return err
			}
			if err := json.Unmarshal(buffer, obj); err != nil {
				return err
			}
		}
		return nil
	}
	{
		filename := "real_test.1.json"

		r1 := NewReal(1.5)
		r1.Alloc(1, 2)
		r2 := NewReal(0.0)

		if err := writeJson(filename, r1); err != nil {
			t.Error(err)
			return
		}
		if err := readJson(filename, r2); err != nil {
			t.Error(err)
			return
		}
		if r1.GetValue() != r2.GetValue() {
			t.Error("test failed")
		}
		os.Remove(filename)
	}
	{
		filename := "real_test.2.json"

		r1 := NewReal(1.5)
		r1.Alloc(1, 2)
		r1.SetDerivative(0, 2.3)
		r2 := NewReal(0.0)

		if err := writeJson(filename, r1); err != nil {
			t.Error(err)
			return
		}
		if err := readJson(filename, r2); err != nil {
			t.Error(err)
			return
		}
		if r1.GetValue() != r2.GetValue() {
			t.Error("test failed")
		}
		if r1.GetDerivative(0) != r2.GetDerivative(0) {
			t.Error("test failed")
		}
		os.Remove(filename)
	}
	{
		filename := "real_test.3.json"

		r1 := NewReal(1.5)
		r1.Alloc(1, 2)
		r1.SetDerivative(0, 2.3)
		r1.SetHessian(0, 0, 3.4)
		r2 := NewReal(0.0)

		if err := writeJson(filename, r1); err != nil {
			t.Error(err)
			return
		}
		if err := readJson(filename, r2); err != nil {
			t.Error(err)
			return
		}
		if r1.GetValue() != r2.GetValue() {
			t.Error("test failed")
		}
		if r1.GetDerivative(0) != r2.GetDerivative(0) {
			t.Error("test failed")
		}
		if r1.GetHessian(0, 0) != r2.GetHessian(0, 0) {
			t.Error("test failed")
		}
		os.Remove(filename)
	}
}

func TestSmoothMax(t *testing.T) {
	x := NewVector(RealType, []float64{-1, 0, 2, 3, 4, 5})
	r := NewReal(0.0)
	t1 := NewReal(0.0)
	t2 := NewReal(0.0)

	r.SmoothMax(x, ConstReal(10), [2]Scalar{t1, t2})

	if math.Abs(r.GetValue()-5) > 1e-4 {
		t.Error("test failed")
	}
}

func TestLogSmoothMax(t *testing.T) {
	x := NewVector(RealType, []float64{0, 1, 10203, 3, 4, 30, 6, 7, 1000, 8, 9, 10})
	r := NewReal(0.0)
	t1 := NewReal(0.0)
	t2 := NewReal(0.0)
	t3 := NewReal(0.0)

	r.LogSmoothMax(x, ConstReal(10), [3]Scalar{t1, t2, t3})

	if math.Abs(r.GetValue()-10203) > 1e-4 {
		t.Error("test failed")
	}
}
