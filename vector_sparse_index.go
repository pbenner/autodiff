/* Copyright (C) 2015-2019 Philipp Benner
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

/* -------------------------------------------------------------------------- */

//import "fmt"

/* -------------------------------------------------------------------------- */

type vectorSparseIndex struct {
	AvlTree
}

type vectorSparseIndexIterator struct {
	AvlIterator
}

/* -------------------------------------------------------------------------- */

func (obj *vectorSparseIndex) indexInsert(i int) {
	obj.AvlTree.Insert(i)
}

func (obj *vectorSparseIndex) indexDelete(i int) {
	obj.AvlTree.Delete(i)
}

func (obj *vectorSparseIndex) indexIterator() vectorSparseIndexIterator {
	return vectorSparseIndexIterator{*obj.AvlTree.Iterator()}
}

func (obj *vectorSparseIndex) indexIteratorFrom(i int) vectorSparseIndexIterator {
	return vectorSparseIndexIterator{*obj.AvlTree.IteratorFrom(i)}
}

func (obj *vectorSparseIndex) indexClone() vectorSparseIndex {
	return vectorSparseIndex{*obj.AvlTree.Clone()}
}

/* -------------------------------------------------------------------------- */

func (obj *vectorSparseIndexIterator) Get() int {
	return obj.AvlIterator.Get().Value
}

func (obj *vectorSparseIndexIterator) Clone() *vectorSparseIndexIterator {
	return &vectorSparseIndexIterator{obj.AvlIterator.Clone()}
}
