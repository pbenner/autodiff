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

#include <boost/math/special_functions/zeta.hpp>

/* -------------------------------------------------------------------------- */

using namespace std;
using namespace boost::math;

/* -------------------------------------------------------------------------- */

void test_zeta_polynomial_series() {
        double j = 2.13;

        for (double i = 0.0; i <= 18.0; i += 0.31) {
                cout << "{"
                     << setw(6)
                     << setprecision( 2) << fixed << i << ", "
                     << setw(6)
                     << setprecision( 2) << fixed << j << ", "
                     << setprecision(20) << fixed << scientific
                     << boost::math::detail::zeta_polynomial_series(i, j, policies::policy<>())
                     << "},"
                     << endl;
        }
}

void test_zeta_odd_integer() {
        double j = 2.13;

        const boost::mpl::true_& t = boost::mpl::true_();

        for (int i = 3; i <= 39; i += 2) {
                cout << "{"
                     << setw(6)
                     << setprecision( 2) << fixed << i << ", "
                     << setw(6)
                     << setprecision( 2) << fixed << j << ", "
                     << setprecision(20) << fixed << scientific
                     << boost::math::detail::zeta_imp_odd_integer(i, j, policies::policy<>(), t)
                     << "},"
                     << endl;
        }
}

void test_zeta() {
        for (double i = -49.70; i <= 50.0; i += 0.35) {
                i = round(i*100)/100;
                cout << "{"
                     << setw(6)
                     << setprecision(2) << fixed << i << ", "
                     << setprecision(20) << fixed << scientific
                     << boost::math::zeta(i)
                     << "},"
                     << endl;
        }
}

int main() {
        //test_zeta_polynomial_series();
        //test_zeta_odd_integer();
        test_zeta();
}
