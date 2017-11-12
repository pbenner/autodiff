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

#include <iostream>
#include <iomanip>

#include <boost/math/special_functions/bessel.hpp>

/* -------------------------------------------------------------------------- */

using namespace std;
using namespace boost::math;

typedef policies::policy<> Policy;
typedef lanczos::undefined_lanczos lanczos_type;
//typedef lanczos::lanczos<double, Policy>::type lanczos_type;

/* -------------------------------------------------------------------------- */

void test_bessel_i() {
        for (double  v = -4.0; v <= 4.0; v += 0.5) {
                for (double x = 0.5; x <= 50; x += 0.5) {
                        cout << "{"
                             << setw(4)
                             << setprecision( 1) << fixed << v << ", "
                             << setw(8)
                             << setprecision( 6) << fixed << x << ", "
                             << setprecision(20) << fixed << scientific
                             << boost::math::cyl_bessel_i(v, x)
                             << "},"
                             << endl;
                }
        }
}

int main() {
        test_bessel_i();
}
