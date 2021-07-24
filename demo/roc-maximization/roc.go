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

package main

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "bufio"
import   "math"
import   "os"
import   "strconv"
import   "strings"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/rprop"
import . "github.com/pbenner/autodiff/statistics"
import . "github.com/pbenner/autodiff/statistics/vectorDistribution"

//import . "github.com/pbenner/pshape/Config"

/* -------------------------------------------------------------------------- */

func readTable(filename string) (Matrix, error) {
    var scanner *bufio.Scanner
    // open file
    f, err := os.Open(filename)
    if err != nil {
      return nil, err
    }
    defer f.Close()

    scanner = bufio.NewScanner(f)

    x := []float64{}
    
    for i := 1; scanner.Scan(); i++ {
        fields := strings.Fields(scanner.Text())
        if len(fields) != 2 {
            return nil, fmt.Errorf("parsing line `%d' failed", i)
        }
        v1, err := strconv.ParseFloat(fields[0], 64)
        if err != nil {
            return nil, fmt.Errorf("parsing line `%d' failed: %v", i, err)
        }
        v2, err := strconv.ParseFloat(fields[1], 64)
        if err != nil {
            return nil, fmt.Errorf("parsing line `%d' failed: %v", i, err)
        }
        x = append(x, v1)
      x = append(x, v2)
    }
    return NewDenseReal64Matrix(x, len(x)/2, 2), nil
}

func writeTable(filename string, x ...Vector) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()

  writer := bufio.NewWriter(f)

  for i := 0; i < x[0].Dim(); i++ {
    for j := 0; j < len(x); j++ {
      if j != 0 {
        fmt.Fprintf(writer, " ")
      }
      fmt.Fprintf(writer, "%f", x[j].Float64At(i))
    }
    fmt.Fprintf(writer, "\n")
  }
  writer.Flush()

  return nil
}

/* -------------------------------------------------------------------------- */

func VectorMax(s Vector) float64 {
  j := 0
  for i := 1; i < s.Dim(); i++ {
    if s.Float64At(i) > s.Float64At(j) {
      j = i
    }
  }
  return s.Float64At(j)
}

func VectorMin(s Vector) float64 {
  j := 0
  for i := 1; i < s.Dim(); i++ {
    if s.Float64At(i) < s.Float64At(j) {
      j = i
    }
  }
  return s.Float64At(j)
}

func RocCurve(test0, test1 Vector, n int, d Scalar) (Vector, Vector, Vector) {
  n1 := test0.Dim()
  n2 := test1.Dim()
  if n1 != n2 {
    panic("invalid arguments")
  }
  min := VectorMin(test0.AppendVector(test1))
  max := VectorMax(test0.AppendVector(test1))
  if set_min := -500.0; min < set_min {
    fmt.Printf("warning: roc minimum insufficient (actual minimum is `%f')\n", min)
  } else {
    min = set_min
  }
  if set_max := 500.0; max > set_max {
    fmt.Printf("warning: roc maximum insufficient (actual maximum is `%f')\n", max)
  } else {
    max = set_max
  }
  step := NewReal64((max-min)/float64(n))
  tmp  := NewReal64(0.0)
  // result
  var thr Vector = NullDenseReal64Vector(0)
  var tpr Vector = NullDenseReal64Vector(0)
  var fpr Vector = NullDenseReal64Vector(0)
  for t := NewReal64(min); t.GetFloat64() < max; t.Add(t, step) {
    tp := NullReal64()
    tn := NullReal64()
    fn := NullReal64()
    fp := NullReal64()
    // groundtruth == 0
    for i := 0; i < test0.Dim(); i++ {
      tmp.Sub(test0.ConstAt(i), t)
      tmp.Sigmoid(tmp, d)
      fp.Add(fp, tmp)
      tn.Add(tn, tmp.Sub(ConstFloat64(1.0), tmp))
    }
    // groundtruth == 1
    for i := 0; i < test1.Dim(); i++ {
      tmp.Sub(test1.ConstAt(i), t)
      tmp.Sigmoid(tmp, d)
      tp.Add(tp, tmp)
      fn.Add(fn, tmp.Sub(ConstFloat64(1.0), tmp))
    }
    thr = thr.AppendScalar(  t.CloneMagicScalar()); tmp.Div(tp, tmp.Add(tp, fn))
    tpr = tpr.AppendScalar(tmp.CloneMagicScalar()); tmp.Div(fp, tmp.Add(fp, tn))
    fpr = fpr.AppendScalar(tmp.CloneMagicScalar())
  }
  return thr, fpr, tpr
}

func AUC(x, y Vector) MagicScalar {
  n1 := x.Dim()
  n2 := y.Dim()
  t  := NullReal64()
  dx := NullReal64()
  dy := NullReal64()
  if n1 != n2 {
    panic("invalid arguments")
  }
  result := NewReal64(0.0)

  for i := 1; i < n1; i++ {
    dx.Abs(t.Sub(x.ConstAt(i), x.ConstAt(i-1)))
    dy.Div(t.Add(y.ConstAt(i), y.ConstAt(i-1)), ConstFloat64(2.0))
    result.Add(result, t.Mul(dx, dy))
  }
  return result
}

/* -------------------------------------------------------------------------- */

func evalModel(d0, d1 *NormalDistribution, x0, x1 ConstMatrix) (Vector, Vector) {
  n0, _ := x0.Dims()
  n1, _ := x1.Dims()

  test0 := NullDenseReal64Vector(n0)
  test1 := NullDenseReal64Vector(n1)

  t0 := NewReal64(0.0)
  t1 := NewReal64(0.0)

  for i := 0; i < n0; i++ {
    d0.LogPdf(t0, x0.ConstRow(i))
    d1.LogPdf(t1, x0.ConstRow(i))

    test0.At(i).Sub(t1, t0)
  }
  for i := 0; i < n1; i++ {
    d0.LogPdf(t0, x1.ConstRow(i))
    d1.LogPdf(t1, x1.ConstRow(i))
    test1.At(i).Sub(t1, t0)
  }
  return test0, test1
}

/* -------------------------------------------------------------------------- */

func createDistributions(variables ConstVector) (*NormalDistribution, *NormalDistribution, error) {
  var err error
  var normal0 *NormalDistribution
  var normal1 *NormalDistribution
  // check variables for NaN values
  for i := 0; i < variables.Dim(); i++ {
    if math.IsNaN(variables.Float64At(i)) {
      return normal0, normal1, err
    }
  }
  // extract parameters
  mu0 := NullDenseReal64Vector(2)
  mu0.Set(variables.ConstSlice(0, 2))
  mu1 := NullDenseReal64Vector(2)
  mu1.Set(variables.ConstSlice(2, 4))
  sigma0 := NullDenseReal64Matrix(2, 2)
  sigma0.At(0,0).Set(variables.ConstAt(4))
  sigma0.At(1,1).Set(variables.ConstAt(5))
  sigma0.At(0,1).Set(variables.ConstAt(6))
  sigma0.At(1,0).Set(variables.ConstAt(6))
  sigma1 := NullDenseReal64Matrix(2, 2)
  sigma1.At(0,0).Set(variables.ConstAt(7))
  sigma1.At(1,1).Set(variables.ConstAt(8))
  sigma1.At(0,1).Set(variables.ConstAt(9))
  sigma1.At(1,0).Set(variables.ConstAt(9))

  // create normal distributions
  normal0, err = NewNormalDistribution(mu0, sigma0)
  if err != nil {
    return normal0, normal1, err
  }
  normal1, err = NewNormalDistribution(mu1, sigma1)
  if err != nil {
    return normal0, normal1, err
  }
  return normal0, normal1, nil
}

func objective(variables ConstVector, x0, x1 Matrix, i *int) (MagicScalar, error) {
  normal0, normal1, err := createDistributions(variables)
  if err != nil {
    return nil, err
  }
  // compute log-likelihood ratio for both data matrices
  test0, test1 := evalModel(normal0, normal1, x0, x1)
  // compute roc curve
  thr, fpr, tpr := RocCurve(test0, test1, 1000, NewReal64(1.0))
  // and the AUC
  result := AUC(fpr, tpr)

  // save distributions
  ExportDistribution(fmt.Sprintf("roc.result/roc.normal0.%d.json", *i), normal0)
  ExportDistribution(fmt.Sprintf("roc.result/roc.normal1.%d.json", *i), normal1)
  // save roc curve
  writeTable(fmt.Sprintf("roc.result/roc.%d.table", *i), thr, fpr, tpr)
  (*i)++

  result.Neg(result)

  return result, nil
}

func main() {
  // load data
  x0, err := readTable("roc.x0.table")
  if err != nil {
    panic(err)
  }
  x1, err := readTable("roc.x1.table")
  if err != nil {
    panic(err)
  }
  // trace
  x_i := NullDenseReal64Vector(0)
  y_i := NullDenseReal64Vector(0)
  // iteration step
  i := 0
  // rprop hook
  hook := func(gradient, step []float64, variables ConstVector, s ConstScalar) bool {
    fmt.Printf("value    : %v\n", s)
    fmt.Printf("variables: %v\n", variables)
    fmt.Printf("gradient : %v\n", gradient)
    fmt.Printf("stepsize : %v\n", step)
    fmt.Printf("\n")
    if i >= 100 {
      return true
    }
    // record value
    x_i = append(x_i, NewReal64(float64(i)))
    y_i = append(y_i, NewReal64(s.GetFloat64()))
    return false
  }
  f := func(variables ConstVector) (MagicScalar, error) {
    return objective(variables, x0, x1, &i)
  }
  // initial value
  v0 := NewDenseFloat64Vector([]float64{-5, 0, -1, 3, 0.5, 0.5, 0, 0.5, 0.5, 0})
  // run rprop
  vn, _ := rprop.Run(f, v0, 0.0001, []float64{1.1, 0.1},
    rprop.Hook{hook},
    rprop.Epsilon{1e-8})
  // save trace
  writeTable("roc.result/roc.table", x_i, y_i)

  fmt.Println(vn)
}
