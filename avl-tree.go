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
  Left   *AvlNode
  Right  *AvlNode
  Parent *AvlNode
}

type AvlTree struct {
  Root *AvlNode
}

/* -------------------------------------------------------------------------- */

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

func (obj *AvlTree) FindNode(i int) *AvlNode {
  node := obj.Root
  for node != nil {
    switch {
    case i < node.Value: node = node.Left
    case i > node.Value: node = node.Right
    default: return node
    }
  }
  return nil
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
  if obj.Root.Value == i {
    obj.Root = nil
    return true
  }
  ok, _ := obj.Root.delete(i, nil)
  return ok
}

func (obj *AvlTree) Iterator() *AvlIterator {
  return NewAvlIterator(obj.Root)
}

func (obj *AvlTree) String() string {
  var buffer bytes.Buffer

  if obj.Root == nil {
    buffer.WriteString("()")
  } else {
    obj.Root.string(&buffer)
  }
  return buffer.String()
}

/* -------------------------------------------------------------------------- */

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
    case i  < parent.Value: parent.setLeft (NewAvlNode(i))
    case i  > parent.Value: parent.setRight(NewAvlNode(i))
    case i == parent.Value: return false, true
    }
    return true, false
  }
  switch {
  case i == obj.Value:
    return false, obj.Balance == 0
  case i  < obj.Value:
    if ok, balanced := obj.Left .insert(i, obj); !ok {
      return ok, obj.Balance == 0
    } else {
      if !balanced {
        switch obj.Balance {
        case  1: obj.Balance =  0
        case  0: obj.Balance = -1
        case -1:
          if obj.Left.Balance == -1 {
            obj.rotateLL()
          } else {
            obj.rotateLR()
          }
        }
      }
    }
  case i  > obj.Value:
    if ok, balanced := obj.Right.insert(i, obj); !ok {
      return ok, true
    } else {
      if !balanced {
        switch obj.Balance {
        case -1: obj.Balance = 0
        case  0: obj.Balance = 1
        case  1:
          if obj.Right.Balance == 1 {
            obj.rotateRR()
          } else {
            obj.rotateRL()
          }
        }
      }
    }
  }
  return true, obj.Balance == 0
}

/* -------------------------------------------------------------------------- */

func (obj *AvlNode) rotateLL() {
  a1 := obj.Left
  a2 := obj.Right

  obj.setLeft (a1.Left)
  obj.setRight(a1)
  a1 .setLeft (a1.Right)
  a1 .setRight(a2)

  obj.Value, a1.Value = a1.Value, obj.Value

  obj.Right.Balance = 0
  obj      .Balance = 0
}

func (obj *AvlNode) rotateLR() {
  a1 := obj.Left
  a2 :=  a1.Right

  a1 .setRight(a2 .Left)
  a2 .setLeft (a2 .Right)
  a2 .setRight(obj.Right)
  obj.setRight(a2)

  obj.Value, a2.Value = a2.Value, obj.Value

  if a2.Balance ==  1 {
    obj.Left .Balance = -1
  } else {
    obj.Left .Balance =  0
  }
  if a2.Balance == -1 {
    obj.Right.Balance =  1
  } else {
    obj.Right.Balance =  0
  }
  obj.Balance = 0
}

func (obj *AvlNode) rotateRR() {
  a1 := obj.Right
  a2 := obj.Left

  obj.setRight(a1.Right)
  obj.setLeft (a1)
  a1 .setRight(a1.Left)
  a1 .setLeft (a2)

  obj.Value, a1.Value = a1.Value, obj.Value

  obj.Left.Balance = 0
  obj     .Balance = 0
}

func (obj *AvlNode) rotateRL() {
  a1 := obj.Right
  a2 :=  a1.Left

  a1 .setLeft ( a2.Right)
  a2 .setRight( a2.Left)
  a2 .setLeft (obj.Left)
  obj.setLeft (a2)

  obj.Value, a2.Value = a2.Value, obj.Value

  if a2.Balance == -1 {
    obj.Right.Balance =  1
  } else {
    obj.Right.Balance =  0
  }
  if a2.Balance ==  1 {
    obj.Left .Balance = -1
  } else {
    obj.Left .Balance =  0
  }
  obj.Balance = 0
}

/* -------------------------------------------------------------------------- */

func (obj *AvlNode) delete(i int, parent *AvlNode) (bool, bool) {
  if obj == nil {
    return false, true
  }
  if i < obj.Value {
    ok, balanced := obj.Left .delete(i, obj)
    if !balanced {
      balanced = obj.balance1()
    }
    return ok, balanced
  }
  if i > obj.Value {
    ok, balanced := obj.Right.delete(i, obj)
    if !balanced {
      balanced = obj.balance2()
    }
    return ok, balanced
  }
  // this node must be deleted
  if obj.Right == nil && obj.Left == nil {
    if i < parent.Value {
      parent.Left  = nil
    } else {
      parent.Right = nil
    }
    return true, false
  }
  if obj.Right == nil {
    if i < parent.Value {
      parent.Left  = obj.Left
    } else {
      parent.Right = obj.Left
    }
    return true, false
  }
  if obj.Left == nil {
    if i < parent.Value {
      parent.Left  = obj.Right
    } else {
      parent.Right = obj.Right
    }
    return true, false
  }
  v_, balanced := obj.Left.deleteRec(obj)
  if !balanced {
    balanced = obj.balance1()
  }
  obj.Value = v_
  return true, balanced
}

func (obj *AvlNode) deleteRec(parent *AvlNode) (int, bool) {
  value := 0
  if obj.Right != nil {
    if v, balanced := obj.Right.deleteRec(obj); !balanced {
      balanced = obj.balance2()
      return value, balanced
    } else {
      value = v
      return value, balanced
    }
  } else {
    value = obj.Value
    if obj.Value > parent.Value {
      parent.setRight(obj.Left)
    } else {
      parent.setLeft (obj.Left)
    }
    return value, false
  }
}

func (obj *AvlNode) balance1() bool {
  switch obj.Balance {
  case -1: obj.Balance = 0
  case  0: obj.Balance = 1
  case  1:
    balance := obj.Right.Balance
    if balance >= 0 {
      obj.rotateRR()
      if balance == 0 {
        obj     .Balance = -1
        obj.Left.Balance =  1
      }
    } else {
      obj.rotateRL()
    }
  }
  return obj.Balance != 0
}

func (obj *AvlNode) balance2() bool {
  fmt.Printf("calling balance2 on `%v'\n", obj)
  switch obj.Balance {
  case  1: obj.Balance =  0
  case  0: obj.Balance = -1
  case -1:
    balance := obj.Left.Balance
    if balance <= 0 {
      obj.rotateLL()
      if balance == 0 {
        obj     .Balance =  1
        obj.Left.Balance = -1
      }
    } else {
      obj.rotateLR()
    }
  }
  return obj.Balance != 0
}

/* -------------------------------------------------------------------------- */

func (obj *AvlNode) string(writer io.Writer) {
  if obj.Left != nil {
    fmt.Fprintf(writer, "(")
    obj.Left.string(writer)
    fmt.Fprintf(writer, "):")
  }
  fmt.Fprintf(writer, "%d", obj.Value)
  if obj.Right != nil {
    fmt.Fprintf(writer, ":(")
    obj.Right.string(writer)
    fmt.Fprintf(writer, ")")
  }
}

/* -------------------------------------------------------------------------- */

type AvlIterator struct {
  node *AvlNode
}

func NewAvlIterator(node *AvlNode) *AvlIterator {
  for node.Left != nil {
    node = node.Left
  }
  r := AvlIterator{}
  r.node = node
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
