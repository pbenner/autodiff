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

#include <boost/math/special_functions/gamma.hpp>

/* -------------------------------------------------------------------------- */

using namespace std;
using namespace boost::math;

typedef policies::policy<> Policy;
typedef lanczos::undefined_lanczos lanczos_type;
//typedef lanczos::lanczos<double, Policy>::type lanczos_type;

/* -------------------------------------------------------------------------- */

void test_upper_gamma_fraction() {
        for (double  a = 1.0; a <= 4.0; a += 0.4) {
                cout << "{"
                     << setw(4)
                     << setprecision( 1) << fixed << a << ", "
                     << setprecision(20) << fixed << scientific
                     << detail::upper_gamma_fraction(a, 10.0, policies::get_epsilon<double, Policy>())
                     << "},"
                     << endl;
        }
}

void test_lower_gamma_series() {
        for (double  a = 1.0; a <= 4.0; a += 0.4) {
                cout << "{"
                     << setw(4)
                     << setprecision( 1) << fixed << a << ", "
                     << setprecision(20) << fixed << scientific
                     << detail::lower_gamma_series(a, 10.0, Policy())
                     << "},"
                     << endl;
        }
}


void test_regularised_gamma_prefix() {
        for (double  a = 1.0; a <= 4.0; a += 0.4) {
                for (double z = 0.05; z <= 4; z += 0.05) {
                        cout << "{"
                             << setw(4)
                             << setprecision( 1) << fixed << a << ", "
                             << setw(8)
                             << setprecision( 6) << fixed << z << ", "
                             << setprecision(20) << fixed << scientific
                             << detail::regularised_gamma_prefix(a, z, Policy(), lanczos_type())
                             << "},"
                             << endl;
                }
        }
        for (double  a = 10.0; a <= 14.0; a += 0.4) {
                for (double z = 0.05; z <= 4; z += 0.05) {
                        cout << "{"
                             << setw(4)
                             << setprecision( 1) << fixed << a << ", "
                             << setw(8)
                             << setprecision( 6) << fixed << z << ", "
                             << setprecision(20) << fixed << scientific
                             << detail::regularised_gamma_prefix(a, z, Policy(), lanczos_type())
                             << "},"
                             << endl;
                }
        }
}

void test_lower_incomplete_gamma() {
        for (double  a = 1.0; a <= 4.0; a += 0.4) {
                for (double z = 0.05; z <= 4; z += 0.05) {
                        cout << "{"
                             << setw(4)
                             << setprecision( 1) << fixed << a << ", "
                             << setw(8)
                             << setprecision( 6) << fixed << z << ", "
                             << setprecision(20) << fixed << scientific
                             << boost::math::gamma_p(a, z)
                             << "},"
                             << endl;
                }
        }
}

int main() {
        test_upper_gamma_fraction();
        //test_lower_gamma_series();
        //test_regularised_gamma_prefix();
        //test_lower_incomplete_gamma();
}
