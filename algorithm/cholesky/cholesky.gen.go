/* Copyright (C) 2015-2020 Philipp Benner
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

//go:generate cpp -P -C -include cholesky_generic.h -nostdinc cholesky.gen.in -o cholesky_generic.go
//go:generate cpp -P -C -include cholesky_float32.h -nostdinc cholesky.gen.in -o cholesky_float32.go
//go:generate cpp -P -C -include cholesky_float64.h -nostdinc cholesky.gen.in -o cholesky_float64.go

/* -------------------------------------------------------------------------- */

package cholesky
