/* Copyright (C) 2019 Philipp Benner
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

import "fmt"
import "bytes"
import "io"

/* -------------------------------------------------------------------------- */

type AvlNode struct {
	Value   int
	Balance int
	Deleted bool
	Left    *AvlNode
	Right   *AvlNode
	Parent  *AvlNode
}

type AvlTree struct {
	Root *AvlNode
}

/* -------------------------------------------------------------------------- */

func NewAvlTree() *AvlTree {
	return &AvlTree{}
}

func NewAvlNode(i int) *AvlNode {
	r := AvlNode{}
	r.Value = i
	return &r
}

/* -------------------------------------------------------------------------- */

func (obj *AvlTree) Emtpy() bool {
	return obj.Root == nil
}

func (obj *AvlTree) Left() AvlTree {
	return AvlTree{obj.Root.Left}
}

func (obj *AvlTree) Right() AvlTree {
	return AvlTree{obj.Root.Right}
}

func (obj *AvlTree) Value() int {
	return obj.Root.Value
}

/* -------------------------------------------------------------------------- */

func (obj *AvlTree) Clone() *AvlTree {
	r := AvlTree{}
	r.Root = obj.Root.clone()
	return &r
}

func (obj *AvlTree) FindNode(i int) *AvlNode {
	node := obj.Root
	for node != nil {
		switch {
		case i < node.Value:
			node = node.Left
		case i > node.Value:
			node = node.Right
		default:
			return node
		}
	}
	return nil
}

func (obj *AvlTree) FindNodeLE(i int) *AvlNode {
	node1 := obj.Root
	node1 = nil
	node2 := obj.Root
	for node2 != nil {
		if i <= node2.Value {
			node1 = node2
		}
		switch {
		case i < node2.Value:
			node2 = node2.Left
		case i > node2.Value:
			node2 = node2.Right
		default:
			return node2
		}
	}
	if node2 == nil {
		return node1
	} else {
		return node2
	}
}

func (obj *AvlTree) Insert(i int) bool {
	if obj.Root == nil {
		obj.Root = NewAvlNode(i)
		return true
	}
	ok, _ := obj.Root.insert(i, nil)
	return ok
}

func (obj *AvlTree) Delete(i int) bool {
	if obj.Root == nil {
		return false
	}
	r, ok, _ := obj.Root.delete(i, nil)
	if ok {
		obj.Root = r
	}
	return ok
}

func (obj *AvlTree) Iterator() *AvlIterator {
	return NewAvlIterator(obj)
}

func (obj *AvlTree) IteratorFrom(i int) *AvlIterator {
	return &AvlIterator{obj.FindNodeLE(i), obj}
}

func (obj AvlTree) String() string {
	var buffer bytes.Buffer

	if obj.Root == nil {
		buffer.WriteString("()")
	} else {
		obj.Root.string(&buffer)
	}
	return buffer.String()
}

/* -------------------------------------------------------------------------- */

func (obj *AvlNode) clone() *AvlNode {
	if obj == nil {
		return nil
	}
	r := AvlNode{}
	r = *obj
	r.setLeft(obj.Left.clone())
	r.setRight(obj.Right.clone())
	return &r
}

func (obj *AvlNode) setLeft(node *AvlNode) {
	obj.Left = node
	if node != nil {
		node.Parent = obj
	}
}

func (obj *AvlNode) setRight(node *AvlNode) {
	obj.Right = node
	if node != nil {
		node.Parent = obj
	}
}

func (obj *AvlNode) insert(i int, parent *AvlNode) (bool, bool) {
	if obj == nil {
		switch {
		case i < parent.Value:
			parent.setLeft(NewAvlNode(i))
		case i > parent.Value:
			parent.setRight(NewAvlNode(i))
		case i == parent.Value:
			return false, true
		}
		return true, false
	}
	switch {
	case i < obj.Value:
		if ok, balanced := obj.Left.insert(i, obj); !ok {
			return ok, balanced
		} else {
			if !balanced {
				switch obj.Balance {
				case 1:
					obj.Balance = 0
					balanced = true
				case 0:
					obj.Balance = -1
				case -1:
					if obj.Left.Balance == -1 {
						obj.rotateLL()
					} else {
						obj.rotateLR()
					}
					balanced = true
				}
			}
			return true, balanced
		}
	case i > obj.Value:
		if ok, balanced := obj.Right.insert(i, obj); !ok {
			return ok, balanced
		} else {
			if !balanced {
				switch obj.Balance {
				case -1:
					obj.Balance = 0
					balanced = true
				case 0:
					obj.Balance = 1
				case 1:
					if obj.Right.Balance == 1 {
						obj.rotateRR()
					} else {
						obj.rotateRL()
					}
					balanced = true
				}
			}
			return true, balanced
		}
	default:
		return false, true
	}
}

/* -------------------------------------------------------------------------- */

func (obj *AvlNode) rotateLL() {
	a1 := obj.Left
	a2 := obj.Right

	obj.setLeft(a1.Left)
	obj.setRight(a1)
	a1.setLeft(a1.Right)
	a1.setRight(a2)

	obj.Value, a1.Value = a1.Value, obj.Value

	obj.Right.Balance = 0
	obj.Balance = 0
}

func (obj *AvlNode) rotateLR() {
	a1 := obj.Left
	a2 := a1.Right

	a1.setRight(a2.Left)
	a2.setLeft(a2.Right)
	a2.setRight(obj.Right)
	obj.setRight(a2)

	obj.Value, a2.Value = a2.Value, obj.Value

	if a2.Balance == 1 {
		obj.Left.Balance = -1
	} else {
		obj.Left.Balance = 0
	}
	if a2.Balance == -1 {
		obj.Right.Balance = 1
	} else {
		obj.Right.Balance = 0
	}
	obj.Balance = 0
}

func (obj *AvlNode) rotateRR() {
	a1 := obj.Right
	a2 := obj.Left

	obj.setRight(a1.Right)
	obj.setLeft(a1)
	a1.setRight(a1.Left)
	a1.setLeft(a2)

	obj.Value, a1.Value = a1.Value, obj.Value

	obj.Left.Balance = 0
	obj.Balance = 0
}

func (obj *AvlNode) rotateRL() {
	a1 := obj.Right
	a2 := a1.Left

	a1.setLeft(a2.Right)
	a2.setRight(a2.Left)
	a2.setLeft(obj.Left)
	obj.setLeft(a2)

	obj.Value, a2.Value = a2.Value, obj.Value

	if a2.Balance == -1 {
		obj.Right.Balance = 1
	} else {
		obj.Right.Balance = 0
	}
	if a2.Balance == 1 {
		obj.Left.Balance = -1
	} else {
		obj.Left.Balance = 0
	}
	obj.Balance = 0
}

/* -------------------------------------------------------------------------- */

func (obj *AvlNode) delete(i int, parent *AvlNode) (*AvlNode, bool, bool) {
	if obj == nil {
		return obj, false, true
	}
	if i < obj.Value {
		r, ok, balanced := obj.Left.delete(i, obj)
		if ok {
			obj.setLeft(r)
		}
		if !balanced {
			balanced = obj.balance1(balanced)
		}
		return obj, ok, balanced
	}
	if i > obj.Value {
		r, ok, balanced := obj.Right.delete(i, obj)
		if ok {
			obj.setRight(r)
		}
		if !balanced {
			balanced = obj.balance2(balanced)
		}
		return obj, ok, balanced
	}
	// this node must be deleted
	obj.Deleted = true
	if obj.Right == nil && obj.Left == nil {
		return nil, true, false
	}
	if obj.Right == nil {
		obj.Left.Parent = nil
		return obj.Left, true, false
	}
	if obj.Left == nil {
		obj.Right.Parent = nil
		return obj.Right, true, false
	}
	n_, balanced := obj.Left.deleteRec(obj)
	obj = obj.replace(n_)
	if !balanced {
		balanced = obj.balance1(balanced)
	}
	return obj, true, balanced
}

func (obj *AvlNode) deleteRec(parent *AvlNode) (*AvlNode, bool) {
	if obj.Right != nil {
		if v, balanced := obj.Right.deleteRec(obj); !balanced {
			balanced = obj.balance2(balanced)
			return v, balanced
		} else {
			return v, balanced
		}
	} else {
		if obj.Value > parent.Value {
			parent.setRight(obj.Left)
		} else {
			parent.setLeft(obj.Left)
		}
		return obj, false
	}
}

func (obj *AvlNode) balance1(balanced bool) bool {
	switch obj.Balance {
	case -1:
		obj.Balance = 0
	case 0:
		obj.Balance = 1
		balanced = true
	case 1:
		balance := obj.Right.Balance
		if balance >= 0 {
			obj.rotateRR()
			if balance == 0 {
				obj.Balance = -1
				obj.Left.Balance = 1
				balanced = true
			}
		} else {
			obj.rotateRL()
		}
	}
	return balanced
}

func (obj *AvlNode) balance2(balanced bool) bool {
	switch obj.Balance {
	case 1:
		obj.Balance = 0
	case 0:
		obj.Balance = -1
		balanced = true
	case -1:
		balance := obj.Left.Balance
		if balance <= 0 {
			obj.rotateLL()
			if balance == 0 {
				obj.Balance = 1
				obj.Right.Balance = -1
				balanced = true
			}
		} else {
			obj.rotateLR()
		}
	}
	return balanced
}

func (obj *AvlNode) replace(node *AvlNode) *AvlNode {
	node.Parent = obj.Parent
	node.Balance = obj.Balance
	node.setRight(obj.Right)
	node.setLeft(obj.Left)
	obj.Parent = nil
	obj.Right = nil
	obj.Left = nil
	return node
}

/* -------------------------------------------------------------------------- */

func (obj *AvlNode) string(writer io.Writer) {
	if obj.Left != nil {
		fmt.Fprintf(writer, "(")
		obj.Left.string(writer)
		fmt.Fprintf(writer, "):")
	}
	fmt.Fprintf(writer, "%d#%d", obj.Value, obj.Balance)
	if obj.Right != nil {
		fmt.Fprintf(writer, ":(")
		obj.Right.string(writer)
		fmt.Fprintf(writer, ")")
	}
}

/* -------------------------------------------------------------------------- */

type AvlIterator struct {
	node *AvlNode
	tree *AvlTree
}

func NewAvlIterator(tree *AvlTree) *AvlIterator {
	node := tree.Root
	if node != nil {
		for node.Left != nil {
			node = node.Left
		}
	}
	r := AvlIterator{}
	r.node = node
	r.tree = tree
	return &r
}

func (obj *AvlIterator) Get() *AvlNode {
	return obj.node
}

func (obj *AvlIterator) Ok() bool {
	return obj.node != nil
}

func (obj *AvlIterator) Next() {
	if obj.node == nil {
		return
	}
	if obj.node.Deleted {
		obj.node = obj.tree.FindNodeLE(obj.node.Value + 1)
		return
	}
	if obj.node.Right != nil {
		// there is a node to the right where we can go
		// further down
		obj.node = obj.node.Right
		for obj.node.Left != nil {
			obj.node = obj.node.Left
		}
	} else {
		// there is no node to the right, so we need to go up
		// => find an ancestor node where this branch is on
		// the left
		for obj.node.Parent != nil && obj.node.Parent.Right != nil && obj.node.Parent.Right == obj.node {
			obj.node = obj.node.Parent
		}
		if obj.node.Parent == nil {
			// we are at the root, stop
			obj.node = nil
		} else {
			// found next node
			obj.node = obj.node.Parent
		}
	}
}

func (obj *AvlIterator) Clone() AvlIterator {
	return AvlIterator{obj.node, obj.tree}
}
