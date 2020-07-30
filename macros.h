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

#define _STR_CONCAT(a,b) a##b
#define  STR_CONCAT(a,b) _STR_CONCAT(a,b)

/* -------------------------------------------------------------------------- */

#define  SCALAR_REFLECT_TYPE STR_CONCAT(SCALAR_NAME, Type)

#define  NEW_SCALAR STR_CONCAT(New,  SCALAR_NAME)
#define NULL_SCALAR STR_CONCAT(Null, SCALAR_NAME)

#define  NEW_VECTOR STR_CONCAT(New,  VECTOR_NAME)
#define NULL_VECTOR STR_CONCAT(Null, VECTOR_NAME)
#define  NIL_VECTOR STR_CONCAT(nil,  VECTOR_NAME)
#define   AS_VECTOR STR_CONCAT(As,   VECTOR_NAME)

#define  NEW_MATRIX STR_CONCAT(New,  MATRIX_NAME)
#define NULL_MATRIX STR_CONCAT(Null, MATRIX_NAME)
#define  NIL_MATRIX STR_CONCAT(nil,  MATRIX_NAME)
#define   AS_MATRIX STR_CONCAT(As,   MATRIX_NAME)
