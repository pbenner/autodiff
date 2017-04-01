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

#include <boost/math/special_functions/polygamma.hpp>

/* -------------------------------------------------------------------------- */

using namespace std;
using namespace boost::math;

/* -------------------------------------------------------------------------- */

void test_polygamma_large() {
        static const char* function = "boost::math::polygamma<%1%>(int, %1%)";

        for (int  i = 4; i <= 8; i++) {
                for (double j = 400; j <= 600; j += 5.0) {
                        cout << "{"
                             << setw(4)
                             << setprecision( 1) << fixed << i << ", "
                             << setw(4)
                             << setprecision( 1) << fixed << j << ", "
                             << setprecision(20) << fixed << scientific
                             << boost::math::detail::polygamma_atinfinityplus(i, j, policies::policy<>(), function)
                             << "},"
                             << endl;
                }
        }
}

void test_polygamma_attransitionplus() {
        static const char* function = "boost::math::polygamma<%1%>(int, %1%)";

        for (int  i = 4; i <= 8; i++) {
                for (double j = 1; j <= 100; j += 0.5) {
                        cout << "{"
                             << setw(4)
                             << setprecision( 1) << fixed << i << ", "
                             << setw(4)
                             << setprecision( 1) << fixed << j << ", "
                             << setprecision(20) << fixed << scientific
                             << boost::math::detail::polygamma_attransitionplus(i, j, policies::policy<>(), function)
                             << "},"
                             << endl;
                }
        }
}

void test_polygamma_near_zero() {
        static const char* function = "boost::math::polygamma<%1%>(int, %1%)";

        for (int  i = 4; i <= 8; i++) {
                for (int j = 1; j <= 100; j += 2) {
                        double k = double(j) / 1000.0;
                        cout << "{"
                             << setw(4)
                             << setprecision( 1) << fixed << i << ", "
                             << setw(8)
                             << setprecision( 6) << fixed << k << ", "
                             << setprecision(20) << fixed << scientific
                             << boost::math::detail::polygamma_nearzero(i, k, policies::policy<>(), function)
                             << "},"
                             << endl;
                }
        }
}

void test_poly_cot_pi() {
        static const char* function = "boost::math::polygamma<%1%>(int, %1%)";

        for (int  i = 4; i <= 30; i++) {
                for (int j = 1; j <= 100; j += 2) {
                        double k = -double(j)/100.0;
                        cout << "{"
                             << setw(4)
                             << setprecision( 1) << fixed << i << ", "
                             << setw(8)
                             << setprecision( 6) << fixed << k << ", "
                             << setprecision(20) << fixed << scientific
                             << boost::math::detail::poly_cot_pi(i, 1.0-k, k, policies::policy<>(), function)
                             << "},"
                             << endl;
                }
        }
}

void test_polygamma() {
        for (int  i = 4; i <= 6; i++) {
                for (int j = -200; j <= 200; j += 5) {
                        double k = double(j)/100.0;
                        if (k == 0 || k == -1 || k == -2) {
                                continue;
                        }
                        cout << "{"
                             << setw(4)
                             << setprecision( 1) << fixed << i << ", "
                             << setw(8)
                             << setprecision( 6) << fixed << k << ", "
                             << setprecision(20) << fixed << scientific
                             << boost::math::polygamma(i, k)
                             << "},"
                             << endl;
                }
        }
}

int main() {
        //test_polygamma_large();
        //test_polygamma_attransitionplus();
        //test_polygamma_near_zero();
        //test_poly_cot_pi();
        test_polygamma();
}
