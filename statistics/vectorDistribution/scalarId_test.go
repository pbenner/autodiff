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
import   "testing"

import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/scalarDistribution"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestScalarId(test *testing.T) {

  d1, _ := scalarDistribution.NewGammaDistribution(NewFloat64(1.0), NewFloat64(2.0))
  d2, _ := scalarDistribution.NewGammaDistribution(NewFloat64(2.0), NewFloat64(3.0))

  id, _ := NewScalarId(d1, d2)

  ExportDistribution("scalarId_test.json", id)

  if d, err := ImportVectorPdf("scalarId_test.json", Float64Type); err != nil {
    test.Error(err)
  } else {
    switch id := d.(type) {
    case *ScalarId:
      if len(id.Distributions) != 2 {
        test.Error("test failed")
      } else {
        switch gamma := id.Distributions[0].(type) {
        case *scalarDistribution.GammaDistribution:
          if gamma.Alpha.GetFloat64() != 1.0 {
            test.Error("test failed")
          }
        default:
          test.Error("test failed")
        }
      }
    default:
      test.Error("test failed")
    }
  }
}
