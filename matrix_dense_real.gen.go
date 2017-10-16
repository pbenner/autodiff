/* Copyright (C) 2015-2017 Philipp Benner
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

//go:generate cpp -P -C -nostdinc -include matrix_dense_real.gen.h matrix_dense_template.in -o matrix_dense_real.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_real.gen.h matrix_dense_template_math.in -o matrix_dense_real_math.go

/* -------------------------------------------------------------------------- */

package autodiff