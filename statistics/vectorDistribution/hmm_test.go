/* Copyright (C) 2017-2020 Philipp Benner
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
import   "math"
import   "os"
import   "testing"

import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/scalarDistribution"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func (obj *Hmm) forwardLogPdf(r Scalar, x ConstVector) error {
  alpha, _, err := obj.ForwardBackward(HmmDataRecord{obj.Edist, x}); if err != nil {
    return err
  }
  t1 := NewScalar(x.ElementType(), 0.0)
  t2 := NewScalar(x.ElementType(), 0.0)
  r.SetFloat64(math.Inf(-1))
  for i := 0; i < obj.M; i++ {
    t1.Set(alpha.At(i, x.Dim()-1))
    r .LogAdd(r, t1, t2)
  }
  return nil
}

func (obj *Hmm) backwardLogPdf(r Scalar, x ConstVector) error {
  _, beta, err := obj.ForwardBackward(HmmDataRecord{obj.Edist, x}); if err != nil {
    return err
  }
  t1 := NewScalar(x.ElementType(), 0.0)
  t2 := NewScalar(x.ElementType(), 0.0)
  r.SetFloat64(math.Inf(-1))
  for i := 0; i < obj.M; i++ {
    obj.Edist[obj.StateMap[i]].LogPdf(t2, x.ConstAt(0))
    t1.Add(beta.At(i, 0), obj.Pi.At(i))
    t1.Add(t1, t2)
    r .LogAdd(r, t1, t2)
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func TestHmm1(test *testing.T) {
  // Hmm definition
  //////////////////////////////////////////////////////////////////////////////
  tr := NewDenseFloat64Matrix(
    []float64{0.7, 0.3, 0.4, 0.6}, 2, 2)

  c1, _ := scalarDistribution.NewCategoricalDistribution(
    NewDenseFloat64Vector([]float64{0.1, 0.9}))
  c2, _ := scalarDistribution.NewCategoricalDistribution(
    NewDenseFloat64Vector([]float64{0.7, 0.3}))

  pi := NewDenseFloat64Vector([]float64{0.6, 0.4})

  hmm, err := NewHmm(pi, tr, nil, []ScalarPdf{c1, c2})
  if err != nil {
    test.Error(err)
  }
  // test conditioning on start and final states
  //////////////////////////////////////////////////////////////////////////////
  {
    hmm := hmm.Clone()
    hmm.SetStartStates([]int{1})
    hmm.SetFinalStates([]int{1})
    {
      x := NewDenseFloat64Vector([]float64{1,1,1})

      r1 := NullFloat64(); hmm.LogPdf(r1, x)
      r2 := NullFloat64(); hmm. forwardLogPdf(r2, x)
      r3 := NullFloat64(); hmm.backwardLogPdf(r3, x)

      if math.Abs(r1.Exp(r1).GetFloat64() - 0.0486) > 1e-4 {
        test.Error("Hmm conditioning test failed")
      }
      if math.Abs(r2.Exp(r2).GetFloat64() - 0.0486) > 1e-4 {
        test.Error("Hmm conditioning test failed")
      }
      if math.Abs(r3.Exp(r3).GetFloat64() - 0.0486) > 1e-4 {
        test.Error("Hmm conditioning test failed")
      }
    }
    {
      x := NewDenseFloat64Vector([]float64{1,1,0,1})

      r1 := NullFloat64(); hmm.LogPdf(r1, x)
      r2 := NullFloat64(); hmm. forwardLogPdf(r2, x)
      r3 := NullFloat64(); hmm.backwardLogPdf(r3, x)

      if math.Abs(r1.Exp(r1).GetFloat64() - 0.016524) > 1e-4 {
        test.Error("Hmm conditioning test failed")
      }
      if math.Abs(r2.Exp(r2).GetFloat64() - 0.016524) > 1e-4 {
        test.Error("Hmm conditioning test failed")
      }
      if math.Abs(r3.Exp(r3).GetFloat64() - 0.016524) > 1e-4 {
        test.Error("Hmm conditioning test failed")
      }
    }
  }
  // test if Pdf is correctly normalized
  //////////////////////////////////////////////////////////////////////////////
  {
    r := NullFloat64()

    hmm := hmm.Clone()
    //hmm.SetStartStates([]int{0})
    //hmm.SetFinalStates([]int{0})
    sum1 := 0.0
    sum2 := 0.0
    sum3 := 0.0
    for _, u := range []float64{0, 1} {
      for _, v := range []float64{0, 1} {
        for _, x := range []float64{0, 1} {
          for _, y := range []float64{0, 1} {
            for _, z := range []float64{0, 1} {
              x := NewDenseFloat64Vector([]float64{u,v,x,y,z})
              hmm.LogPdf(r, x);         sum1 += r.Exp(r).GetFloat64()
              hmm. forwardLogPdf(r, x); sum2 += r.Exp(r).GetFloat64()
              hmm.backwardLogPdf(r, x); sum3 += r.Exp(r).GetFloat64()
            }
          }
        }
      }
    }
    if math.Abs(sum1-1.0) > 1e-8 {
      test.Error("Hmm summation test failed")
    }
    if math.Abs(sum2-1.0) > 1e-8 {
      test.Error("Hmm summation test failed")
    }
    if math.Abs(sum3-1.0) > 1e-8 {
      test.Error("Hmm summation test failed")
    }
  }
  // test if Pdf is correctly normalized with conditioning
  //////////////////////////////////////////////////////////////////////////////
  {
    r := NullFloat64()

    hmm := hmm.Clone()
    hmm.SetStartStates([]int{0})
    hmm.SetFinalStates([]int{0})
    sum1 := 0.0
    sum2 := 0.0
    sum3 := 0.0
    for _, u := range []float64{0, 1} {
      for _, v := range []float64{0, 1} {
        for _, x := range []float64{0, 1} {
          for _, y := range []float64{0, 1} {
            for _, z := range []float64{0, 1} {
              x := NewDenseFloat64Vector([]float64{u,v,x,y,z})
              hmm.LogPdf(r, x);         sum1 += r.Exp(r).GetFloat64()
              hmm. forwardLogPdf(r, x); sum2 += r.Exp(r).GetFloat64()
              hmm.backwardLogPdf(r, x); sum3 += r.Exp(r).GetFloat64()
            }
          }
        }
      }
    }
    if math.Abs(sum1-1.0) > 1e-8 {
      test.Error("Hmm summation test failed")
    }
    if math.Abs(sum2-1.0) > 1e-8 {
      test.Error("Hmm summation test failed")
    }
    if math.Abs(sum3-1.0) > 1e-8 {
      test.Error("Hmm summation test failed")
    }
  }
  // test forward/backward algorithm
  //////////////////////////////////////////////////////////////////////////////
  {
    x := NewDenseFloat64Vector([]float64{1,1,1,1,1,1,0,0,1,0})

    hmm := hmm.Clone()
    //hmm.SetStartStates([]int{0})
    //hmm.SetFinalStates([]int{0})

    r1 := NullFloat64(); hmm.LogPdf(r1, x)
    r2 := NullFloat64(); hmm. forwardLogPdf(r2, x)
    r3 := NullFloat64(); hmm.backwardLogPdf(r3, x)

    if math.Abs(r2.Sub(r1, r2).GetFloat64()) > 1e-8 {
      test.Error("Hmm forward test failed")
    }
    if math.Abs(r3.Sub(r1, r3).GetFloat64()) > 1e-8 {
      test.Error("Hmm backward test failed")
    }
  }
  // test forward/backward algorithm with conditioning
  //////////////////////////////////////////////////////////////////////////////
  {
    x := NewDenseFloat64Vector([]float64{1,1,1,1,1,1,0,0,1,0})

    hmm := hmm.Clone()
    hmm.SetStartStates([]int{0})
    hmm.SetFinalStates([]int{0})

    r1 := NullFloat64(); hmm.LogPdf(r1, x)
    r2 := NullFloat64(); hmm. forwardLogPdf(r2, x)
    r3 := NullFloat64(); hmm.backwardLogPdf(r3, x)

    if math.Abs(r2.Sub(r1, r2).GetFloat64()) > 1e-8 {
      test.Error("Hmm forward test failed")
    }
    if math.Abs(r3.Sub(r1, r3).GetFloat64()) > 1e-8 {
      test.Error("Hmm backward test failed")
    }
  }
  // test viterbi algorithm
  //////////////////////////////////////////////////////////////////////////////
  {
    x := NewDenseFloat64Vector([]float64{1,1,1,1,1,1,0,0,1,0})

    r    := []int{0, 0, 0, 0, 0, 0, 1, 1, 0, 1}
    v, _ := hmm.Viterbi(x)

    for i := 0; i < len(r); i++ {
      if r[i] != v[i] {
        test.Errorf("Viterbi test failed at position `%d'", i)
      }
    }
  }
  // test extremes
  //////////////////////////////////////////////////////////////////////////////
  {
    x  := NewDenseFloat64Vector([]float64{1})

    hmm := hmm.Clone()

    r1 := NullFloat64(); hmm.LogPdf(r1, x)
    r2 := NullFloat64(); hmm. forwardLogPdf(r2, x)
    r3 := NullFloat64(); hmm.backwardLogPdf(r3, x)

    if math.Abs(r1.Exp(r1).GetFloat64() - 0.66) > 1e-4 {
      test.Error("Hmm test failed")
    }
    if math.Abs(r2.Exp(r2).GetFloat64() - 0.66) > 1e-4 {
      test.Error("Hmm test failed")
    }
    if math.Abs(r3.Exp(r3).GetFloat64() - 0.66) > 1e-4 {
      test.Error("Hmm test failed")
    }
  }
  // test extremes
  //////////////////////////////////////////////////////////////////////////////
  {
    x := NewDenseFloat64Vector([]float64{1})

    hmm := hmm.Clone()
    hmm.SetStartStates([]int{1})
    hmm.SetFinalStates([]int{1})

    r1 := NullFloat64(); hmm.LogPdf(r1, x)
    r2 := NullFloat64(); hmm. forwardLogPdf(r2, x)
    r3 := NullFloat64(); hmm.backwardLogPdf(r3, x)

    if math.Abs(r1.Exp(r1).GetFloat64() - 0.3) > 1e-4 {
      test.Error("Hmm test failed")
    }
    if math.Abs(r2.Exp(r2).GetFloat64() - 0.3) > 1e-4 {
      test.Error("Hmm test failed")
    }
    if math.Abs(r3.Exp(r3).GetFloat64() - 0.3) > 1e-4 {
      test.Error("Hmm test failed")
    }
  }
}

func TestHmm2(test *testing.T) {

  t  := NewFloat64(0.0)
  pi := NewDenseFloat64Vector([]float64{1, 2})
  tr := NewDenseFloat64Matrix([]float64{1,2,3,4}, 2, 2)

  d1, _ := scalarDistribution.NewGammaDistribution(NewFloat64(1.0), NewFloat64(2.0))
  d2, _ := scalarDistribution.NewGammaDistribution(NewFloat64(2.0), NewFloat64(3.0))

  hmm1, _ := NewHmm(pi, tr, nil, []ScalarPdf{d1, d2})

  if err := ExportDistribution("hmm_test.json", hmm1); err != nil {
    test.Error("test failed")
  }
  if hmm2, err := ImportVectorPdf("hmm_test.json", Float64Type); err != nil {
    test.Error(err)
  } else {
    par1 := hmm1.GetParameters()
    par2 := hmm2.GetParameters()
    if t.Vnorm(par1.VsubV(par1, par2)).GetFloat64() > 1e-10 {
      test.Error("test failed")
    }
  }
  os.Remove("hmm_test.json")
}

func TestHmm3(test *testing.T) {
  // Hmm definition
  //////////////////////////////////////////////////////////////////////////////
  tr := NewDenseFloat64Matrix(
    []float64{0.7, 0.3, 0.4, 0.6}, 2, 2)

  pi := NewDenseFloat64Vector([]float64{0.6, 0.4})

  stateMap := []int{0, 0}

  hmm1, err := NewHmm(pi, tr, nil, nil)
  if err != nil {
    test.Error(err)
  }

  hmm2, err := NewHmm(pi, tr, stateMap, nil)
  if err != nil {
    test.Error(err)
  }

  if len(hmm1.Edist) != 2 {
    test.Error("test failed")
  }
  if len(hmm2.Edist) != 1 {
    test.Error("test failed")
  }
}
