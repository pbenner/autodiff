/* Copyright (C) 2017 Philipp Benner
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

package generic

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"
import "reflect"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

// tree for describing the structure of the
// hierarchical hmm
type HmmNode struct {
	Children []HmmNode
	States   [2]int
}

func NewHmmNode(children ...HmmNode) HmmNode {
	n := len(children)
	from := children[0].States[0]
	to := children[n-1].States[1]
	return HmmNode{children, [2]int{from, to}}
}

func NewHmmLeaf(from, to int) HmmNode {
	return HmmNode{nil, [2]int{from, to}}
}

/* -------------------------------------------------------------------------- */

func (node HmmNode) checkRec(states [][2]int) [][2]int {
	for i := 0; i < len(node.Children); i++ {
		states = node.Children[i].checkRec(states)
	}
	if node.Children == nil {
		states = append(states, node.States)
	}
	return states
}

func (node HmmNode) Check(n int) bool {
	states := node.checkRec(nil)
	if len(states) == 0 {
		return false
	}
	// check that for every state, from is smaller than to
	for i := 0; i < len(states); i++ {
		if states[i][0] >= states[i][1] {
			return false
		}
	}
	// check that there is no gap in the set of states
	if states[0][0] != 0 {
		return false
	}
	r := states[0][1]
	for i := 1; i < len(states); i++ {
		if states[i][0] != r {
			return false
		}
		r = states[i][1]
	}
	if r != n {
		return false
	}
	return true
}

/* -------------------------------------------------------------------------- */

func (node HmmNode) ExportConfig() interface{} {
	r := []interface{}{}
	for i := 0; i < len(node.Children); i++ {
		r = append(r, node.Children[i].ExportConfig().([]interface{})...)
	}
	if len(node.Children) == 0 {
		r = append(r, node.States)
	}
	return r
}

func (node *HmmNode) ImportConfig(v interface{}) bool {
	switch reflect.TypeOf(v).Kind() {
	case reflect.Slice:
		s := reflect.ValueOf(v)
		if s.Len() == 2 &&
			// parse leaf
			(reflect.TypeOf(s.Index(0).Elem().Interface()).Kind() == reflect.Float64) &&
			(reflect.TypeOf(s.Index(1).Elem().Interface()).Kind() == reflect.Float64) {
			node.States[0] = int(reflect.ValueOf(s.Index(0).Elem().Interface()).Float())
			node.States[1] = int(reflect.ValueOf(s.Index(1).Elem().Interface()).Float())
		} else {
			// parse internal node
			for i := 0; i < s.Len(); i++ {
				child := HmmNode{}
				if ok := child.ImportConfig(s.Index(i).Elem().Interface()); !ok {
					return false
				}
				node.Children = append(node.Children, child)
			}
		}
		return true
	}
	return false
}

/* -------------------------------------------------------------------------- */

func (node HmmNode) StdInit(tr Matrix, v []float64) error {
	if len(v) == 0 {
		return fmt.Errorf("insufficient number of values")
	}
	for i := node.States[0]; i < node.States[1]; i++ {
		for j := node.States[0]; j < node.States[1]; j++ {
			tr.At(i, j).SetValue(v[0])
		}
	}
	for k := 0; k < len(node.Children); k++ {
		if err := node.Children[k].StdInit(tr, v[1:]); err != nil {
			return err
		}
	}
	return nil
}

/* -------------------------------------------------------------------------- */

type HhmmTransitionMatrix struct {
	Matrix
	Tree HmmNode
}

func NewHhmmTransitionMatrix(tr Matrix, tree HmmNode, isLog bool) (HhmmTransitionMatrix, error) {
	tr = tr.CloneMatrix()
	// log-transform all probabilities
	if !isLog {
		tr.Map(func(x Scalar) { x.Log(x) })
	}
	r := HhmmTransitionMatrix{tr, tree}
	if err := r.Normalize(); err != nil {
		return HhmmTransitionMatrix{}, err
	}
	return r, nil
}

func (obj HhmmTransitionMatrix) GetMatrix() Matrix {
	return obj.Matrix
}

func (obj HhmmTransitionMatrix) Normalize() error {
	obj.normalize(obj.Tree)
	return nil
}

func (obj HhmmTransitionMatrix) CloneTransitionMatrix() TransitionMatrix {
	return HhmmTransitionMatrix{
		obj.Matrix.CloneMatrix(),
		obj.Tree}
}

/* -------------------------------------------------------------------------- */

func (obj HhmmTransitionMatrix) renormalizeSubmatrix(rfrom, rto, cfrom, cto int, c Scalar) {
	tr := obj.Matrix
	for i := rfrom; i < rto; i++ {
		for j := cfrom; j < cto; j++ {
			tr.At(i, j).Sub(tr.At(i, j), c)
		}
	}
}

func (obj HhmmTransitionMatrix) normalizeInt(rfrom, rto, cfrom, cto int, lambda Scalar) Scalar {
	tr := obj.Matrix
	t := tr.ElementType()
	// r = sum lambda
	r := lambda.CloneScalar()
	t1 := NewScalar(t, math.Inf(-1))
	t2 := NewScalar(t, math.Inf(-1))
	// sum over all values in the given submatrix (sum xi)
	for i := rfrom; i < rto; i++ {
		for j := cfrom; j < cto; j++ {
			t1.LogAdd(t1, tr.At(i, j), t2)
		}
	}
	// r = (sum xi)/(sum lambda)
	r.Sub(t1, r)
	// divide by the number of columns
	// t1 = (sum xi)/(n sum lambda)
	t1.Sub(r, ConstReal(math.Log(float64(cto-cfrom))))
	for i := rfrom; i < rto; i++ {
		for j := cfrom; j < cto; j++ {
			tr.At(i, j).Set(t1)
		}
	}
	return r
}

func (obj HhmmTransitionMatrix) normalizeLeaf(node HmmNode) Scalar {
	tr := obj.Matrix
	t := tr.ElementType()
	r := NewScalar(t, math.Inf(-1))
	t1 := NewScalar(t, math.Inf(-1))
	t2 := NewScalar(t, math.Inf(-1))
	// row/column ranges
	from := node.States[0]
	to := node.States[1]
	// loop over rows
	for i := from; i < to; i++ {
		t1.SetValue(math.Inf(-1))
		// sum over values in row i
		for j := from; j < to; j++ {
			t1.LogAdd(t1, tr.At(i, j), t2)
		}
		// normalize values in row i
		for j := from; j < to; j++ {
			tr.At(i, j).Sub(tr.At(i, j), t1)
		}
		// sum up normalization constant
		r.LogAdd(r, t1, t2)
	}
	return r
}

func (obj HhmmTransitionMatrix) normalize(node HmmNode) Scalar {
	tr := obj.Matrix
	t := tr.ElementType()
	c := NewScalar(t, 0.0)
	r := NewScalar(t, math.Inf(-1))
	t2 := NewScalar(t, math.Inf(-1))
	// if this is a leaf, we're done
	if n := len(node.Children); n == 0 {
		return obj.normalizeLeaf(node)
	} else {
		from := node.Children[0].States[0]
		to := node.Children[n-1].States[1]
		// this is an internal node
		for i := 0; i < n; i++ {
			c.SetValue(0.0)
			// normalize children first
			// t1 = sum lambda
			t1 := obj.normalize(node.Children[i])
			// normalize transitions between child i and all other
			// children
			rfrom := node.Children[i].States[0]
			rto := node.Children[i].States[1]
			for j := 0; j < n; j++ {
				if i == j {
					continue
				}
				cfrom := node.Children[j].States[0]
				cto := node.Children[j].States[1]
				c.LogAdd(c,
					obj.normalizeInt(rfrom, rto, cfrom, cto, t1),
					t2)
			}
			// renormalize submatrix
			obj.renormalizeSubmatrix(rfrom, rto, from, to, c)
			// sum lambda = c sum lambda'
			t1.Add(t1, c)
			// sum up normalization constants
			r.LogAdd(r, t1, t2)
		}
		return r
	}
}
