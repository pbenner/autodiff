/* -*- mode: go; -*-
 *
 * Copyright (C) 2015-2020 Philipp Benner
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

package autodiff

//go:generate cpp -P -C -nostdinc -include matrix_dense_float32.h matrix_dense_template.in      -o matrix_dense_float32.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_float32.h matrix_dense_template_math.in -o matrix_dense_float32_math.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_float64.h matrix_dense_template.in      -o matrix_dense_float64.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_float64.h matrix_dense_template_math.in -o matrix_dense_float64_math.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_int16.h matrix_dense_template.in      -o matrix_dense_int16.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_int16.h matrix_dense_template_math.in -o matrix_dense_int16_math.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_int32.h matrix_dense_template.in      -o matrix_dense_int32.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_int32.h matrix_dense_template_math.in -o matrix_dense_int32_math.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_int64.h matrix_dense_template.in      -o matrix_dense_int64.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_int64.h matrix_dense_template_math.in -o matrix_dense_int64_math.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_int8.h matrix_dense_template.in      -o matrix_dense_int8.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_int8.h matrix_dense_template_math.in -o matrix_dense_int8_math.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_int.h matrix_dense_template.in      -o matrix_dense_int.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_int.h matrix_dense_template_math.in -o matrix_dense_int_math.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_real32.h matrix_dense_real_template.in -o matrix_dense_real32.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_real32.h matrix_dense_real_template_math.in -o matrix_dense_real32_math.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_real64.h matrix_dense_real_template.in -o matrix_dense_real64.go
//go:generate cpp -P -C -nostdinc -include matrix_dense_real64.h matrix_dense_real_template_math.in -o matrix_dense_real64_math.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_float32.h matrix_sparse_template.in      -o matrix_sparse_float32.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_float32.h matrix_sparse_template_math.in -o matrix_sparse_float32_math.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_float64.h matrix_sparse_template.in      -o matrix_sparse_float64.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_float64.h matrix_sparse_template_math.in -o matrix_sparse_float64_math.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_int16.h matrix_sparse_template.in      -o matrix_sparse_int16.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_int16.h matrix_sparse_template_math.in -o matrix_sparse_int16_math.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_int32.h matrix_sparse_template.in      -o matrix_sparse_int32.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_int32.h matrix_sparse_template_math.in -o matrix_sparse_int32_math.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_int64.h matrix_sparse_template.in      -o matrix_sparse_int64.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_int64.h matrix_sparse_template_math.in -o matrix_sparse_int64_math.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_int8.h matrix_sparse_template.in      -o matrix_sparse_int8.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_int8.h matrix_sparse_template_math.in -o matrix_sparse_int8_math.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_int.h matrix_sparse_template.in      -o matrix_sparse_int.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_int.h matrix_sparse_template_math.in -o matrix_sparse_int_math.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_real32.h matrix_sparse_real_template.in      -o matrix_sparse_real32.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_real32.h matrix_sparse_real_template_math.in -o matrix_sparse_real32_math.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_real64.h matrix_sparse_real_template.in      -o matrix_sparse_real64.go
//go:generate cpp -P -C -nostdinc -include matrix_sparse_real64.h matrix_sparse_real_template_math.in -o matrix_sparse_real64_math.go
//go:generate cpp -P -C -nostdinc -include scalar_const_float32.h scalar_const_template.in -o scalar_const_float32.go
//go:generate cpp -P -C -nostdinc -include scalar_const_float64.h scalar_const_template.in -o scalar_const_float64.go
//go:generate cpp -P -C -nostdinc -include scalar_const_int16.h scalar_const_template.in -o scalar_const_int16.go
//go:generate cpp -P -C -nostdinc -include scalar_const_int32.h scalar_const_template.in -o scalar_const_int32.go
//go:generate cpp -P -C -nostdinc -include scalar_const_int64.h scalar_const_template.in -o scalar_const_int64.go
//go:generate cpp -P -C -nostdinc -include scalar_const_int8.h scalar_const_template.in -o scalar_const_int8.go
//go:generate cpp -P -C -nostdinc -include scalar_const_int.h scalar_const_template.in -o scalar_const_int.go
//go:generate cpp -P -C -nostdinc -include scalar_float32.h scalar_template.in               -o scalar_float32.go
//go:generate cpp -P -C -nostdinc -include scalar_float32.h scalar_template_math.in          -o scalar_float32_math.go
//go:generate cpp -P -C -nostdinc -include scalar_float32.h scalar_template_math_concrete.in -o scalar_float32_math_concrete.go
//go:generate cpp -P -C -nostdinc -include scalar_float64.h scalar_template.in               -o scalar_float64.go
//go:generate cpp -P -C -nostdinc -include scalar_float64.h scalar_template_math.in          -o scalar_float64_math.go
//go:generate cpp -P -C -nostdinc -include scalar_float64.h scalar_template_math_concrete.in -o scalar_float64_math_concrete.go
//go:generate cpp -P -C -nostdinc -include scalar_int16.h scalar_template.in               -o scalar_int16.go
//go:generate cpp -P -C -nostdinc -include scalar_int16.h scalar_template_math.in          -o scalar_int16_math.go
//go:generate cpp -P -C -nostdinc -include scalar_int16.h scalar_template_math_concrete.in -o scalar_int16_math_concrete.go
//go:generate cpp -P -C -nostdinc -include scalar_int32.h scalar_template.in               -o scalar_int32.go
//go:generate cpp -P -C -nostdinc -include scalar_int32.h scalar_template_math.in          -o scalar_int32_math.go
//go:generate cpp -P -C -nostdinc -include scalar_int32.h scalar_template_math_concrete.in -o scalar_int32_math_concrete.go
//go:generate cpp -P -C -nostdinc -include scalar_int64.h scalar_template.in               -o scalar_int64.go
//go:generate cpp -P -C -nostdinc -include scalar_int64.h scalar_template_math.in          -o scalar_int64_math.go
//go:generate cpp -P -C -nostdinc -include scalar_int64.h scalar_template_math_concrete.in -o scalar_int64_math_concrete.go
//go:generate cpp -P -C -nostdinc -include scalar_int8.h scalar_template.in               -o scalar_int8.go
//go:generate cpp -P -C -nostdinc -include scalar_int8.h scalar_template_math.in          -o scalar_int8_math.go
//go:generate cpp -P -C -nostdinc -include scalar_int8.h scalar_template_math_concrete.in -o scalar_int8_math_concrete.go
//go:generate cpp -P -C -nostdinc -include scalar_int.h scalar_template.in               -o scalar_int.go
//go:generate cpp -P -C -nostdinc -include scalar_int.h scalar_template_math.in          -o scalar_int_math.go
//go:generate cpp -P -C -nostdinc -include scalar_int.h scalar_template_math_concrete.in -o scalar_int_math_concrete.go
//go:generate cpp -P -C -nostdinc -include scalar_real32.h scalar_real_template.in               -o scalar_real32.go
//go:generate cpp -P -C -nostdinc -include scalar_real32.h scalar_real_template_derivative.in    -o scalar_real32_derivative.go
//go:generate cpp -P -C -nostdinc -include scalar_real32.h scalar_real_template_math.in          -o scalar_real32_math.go
//go:generate cpp -P -C -nostdinc -include scalar_real32.h scalar_real_template_math_concrete.in -o scalar_real32_math_concrete.go
//go:generate cpp -P -C -nostdinc -include scalar_real64.h scalar_real_template.in               -o scalar_real64.go
//go:generate cpp -P -C -nostdinc -include scalar_real64.h scalar_real_template_derivative.in    -o scalar_real64_derivative.go
//go:generate cpp -P -C -nostdinc -include scalar_real64.h scalar_real_template_math.in          -o scalar_real64_math.go
//go:generate cpp -P -C -nostdinc -include scalar_real64.h scalar_real_template_math_concrete.in -o scalar_real64_math_concrete.go
//go:generate cpp -P -C -nostdinc -include vector_dense_float32.h vector_dense_template.in      -o vector_dense_float32.go
//go:generate cpp -P -C -nostdinc -include vector_dense_float32.h vector_dense_template_math.in -o vector_dense_float32_math.go
//go:generate cpp -P -C -nostdinc -include vector_dense_float64.h vector_dense_template.in      -o vector_dense_float64.go
//go:generate cpp -P -C -nostdinc -include vector_dense_float64.h vector_dense_template_math.in -o vector_dense_float64_math.go
//go:generate cpp -P -C -nostdinc -include vector_dense_int16.h vector_dense_template.in      -o vector_dense_int16.go
//go:generate cpp -P -C -nostdinc -include vector_dense_int16.h vector_dense_template_math.in -o vector_dense_int16_math.go
//go:generate cpp -P -C -nostdinc -include vector_dense_int32.h vector_dense_template.in      -o vector_dense_int32.go
//go:generate cpp -P -C -nostdinc -include vector_dense_int32.h vector_dense_template_math.in -o vector_dense_int32_math.go
//go:generate cpp -P -C -nostdinc -include vector_dense_int64.h vector_dense_template.in      -o vector_dense_int64.go
//go:generate cpp -P -C -nostdinc -include vector_dense_int64.h vector_dense_template_math.in -o vector_dense_int64_math.go
//go:generate cpp -P -C -nostdinc -include vector_dense_int8.h vector_dense_template.in      -o vector_dense_int8.go
//go:generate cpp -P -C -nostdinc -include vector_dense_int8.h vector_dense_template_math.in -o vector_dense_int8_math.go
//go:generate cpp -P -C -nostdinc -include vector_dense_int.h vector_dense_template.in      -o vector_dense_int.go
//go:generate cpp -P -C -nostdinc -include vector_dense_int.h vector_dense_template_math.in -o vector_dense_int_math.go
//go:generate cpp -P -C -nostdinc -include vector_dense_real32.h vector_dense_real_template.in      -o vector_dense_real32.go
//go:generate cpp -P -C -nostdinc -include vector_dense_real32.h vector_dense_real_template_math.in -o vector_dense_real32_math.go
//go:generate cpp -P -C -nostdinc -include vector_dense_real64.h vector_dense_real_template.in      -o vector_dense_real64.go
//go:generate cpp -P -C -nostdinc -include vector_dense_real64.h vector_dense_real_template_math.in -o vector_dense_real64_math.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_const_float32.h vector_sparse_const_template.in -o vector_sparse_const_float32.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_const_float64.h vector_sparse_const_template.in -o vector_sparse_const_float64.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_const_int16.h vector_sparse_const_template.in -o vector_sparse_const_int16.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_const_int32.h vector_sparse_const_template.in -o vector_sparse_const_int32.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_const_int64.h vector_sparse_const_template.in -o vector_sparse_const_int64.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_const_int8.h vector_sparse_const_template.in -o vector_sparse_const_int8.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_const_int.h vector_sparse_const_template.in -o vector_sparse_const_int.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_float32.h vector_sparse_template.in      -o vector_sparse_float32.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_float32.h vector_sparse_template_math.in -o vector_sparse_float32_math.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_float64.h vector_sparse_template.in      -o vector_sparse_float64.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_float64.h vector_sparse_template_math.in -o vector_sparse_float64_math.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_int16.h vector_sparse_template.in      -o vector_sparse_int16.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_int16.h vector_sparse_template_math.in -o vector_sparse_int16_math.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_int32.h vector_sparse_template.in      -o vector_sparse_int32.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_int32.h vector_sparse_template_math.in -o vector_sparse_int32_math.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_int64.h vector_sparse_template.in      -o vector_sparse_int64.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_int64.h vector_sparse_template_math.in -o vector_sparse_int64_math.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_int8.h vector_sparse_template.in      -o vector_sparse_int8.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_int8.h vector_sparse_template_math.in -o vector_sparse_int8_math.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_int.h vector_sparse_template.in      -o vector_sparse_int.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_int.h vector_sparse_template_math.in -o vector_sparse_int_math.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_real32.h vector_sparse_real_template.in      -o vector_sparse_real32.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_real32.h vector_sparse_real_template_math.in -o vector_sparse_real32_math.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_real64.h vector_sparse_real_template.in      -o vector_sparse_real64.go
//go:generate cpp -P -C -nostdinc -include vector_sparse_real64.h vector_sparse_real_template_math.in -o vector_sparse_real64_math.go
