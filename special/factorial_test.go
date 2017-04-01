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

package special

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"
import "testing"

/* -------------------------------------------------------------------------- */

func TestFactorial(t *testing.T) {
  r := [][]float64{
    {   0,                                                1},
    {   1,                                                1},
    {   2,                                                2},
    {   3,                                                6},
    {   4,                                               24},
    {   5,                                              120},
    {   6,                                              720},
    {   7,                                             5040},
    {   8,                                            40320},
    {   9,                                           362880},
    {  10,                                          3628800},
    {  11,                                         39916800},
    {  12,                                        479001600},
    {  13,                                       6227020800},
    {  14,                                      87178291200},
    {  15,                                    1307674368000},
    {  16,                                   20922789888000},
    {  17,                                  355687428096000},
    {  18,                                 6402373705728000},
    {  19,                               121645100408832000},
    {  20,                              2432902008176640000},
    {  21,                             51090942171709440000},
    {  22,                           1124000727777607680000},
    {  23,                          25852016738884978212864},
    {  24,                         620448401733239409999872},
    {  25,                       15511210043330986055303168},
    {  26,                      403291461126605650322784256},
    {  27,                    10888869450418351940239884288},
    {  28,                   304888344611713871918902804480},
    {  29,                  8841761993739701898620088352768},
    {  30,                265252859812191068217601719009280},
    {  31,               8222838654177922430198509928972288},
    {  32,             263130836933693517766352317727113216},
    {  33,            8683317618811885938715673895318323200},
    {  34,          295232799039604157334081533963162091520},
    {  35,        10333147966386145431134989962796349784064},
    {  36,       371993326789901254863672752494735387525120},
    {  37,     13763753091226345578872114833606270345281536},
    {  38,    523022617466601117141859892252474974331207680},
    {  39,  20397882081197444123129673397696887153724751872} }

  for i := 0; i < len(r); i++ {
    if math.Abs(Factorial(int(r[i][0])) - r[i][1]) > math.Pow(8, float64(i)) {
      t.Errorf("Factorial() failed for `%f'\n", r[i][0])
    }
  }
}
