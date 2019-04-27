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

type AvlNode struct {
  Value   int
  Balance int
  Left   *AvlNode
  Right  *AvlNode
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

func (obj AvlTree) Emtpy() bool {
  return obj.Root == nil
}

func (obj AvlTree) Left() AvlTree {
  return AvlTree{obj.Root.Left}
}

func (obj AvlTree) Right() AvlTree {
  return AvlTree{obj.Root.Right}
}

func (obj AvlTree) Value() int {
  return obj.Root.Value
}

/* -------------------------------------------------------------------------- */

func (obj AvlTree) FindNode(i int) *AvlNode {
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

func (obj AvlTree) Insert(i int) bool {
  if obj.Root == nil {
    obj.Root = NewAvlNode(i)
    return true
  }
  ok, _ := obj.Root.insert(i, nil)
  return ok
}

/* -------------------------------------------------------------------------- */

func (obj *AvlNode) insert(i int, parent *AvlNode) (bool, bool) {
  if obj == nil {
    switch {
    case i  < obj.Value: parent.Left  = NewAvlNode(i)
    case i  > obj.Value: parent.Right = NewAvlNode(i)
    case i == obj.Value: return false, true
    }
    return true, false
  }
  switch {
  case i == obj.Value:
    return false, true
  case i  < obj.Value:
    if ok, balanced := obj.Left .insert(i, obj); !ok {
      return ok, true
    } else {
      if !balanced {
        if obj.Balance == 1 {
          obj.Balance = 0
        } else
        if obj.Balance == 0 {
          obj.Balance = -1
        } else {
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
        if obj.Balance == -1 {
          obj.Balance = 0
        } else
        if obj.Balance ==  0 {
          obj.Balance = 1
        } else {
          if obj.Right.Balance == 1 {
            obj.rotateRR()
          } else {
            obj.rotateRL()
          }
        }
      }
    }
  }
  return true, true
}

/* -------------------------------------------------------------------------- */

func (obj *AvlNode) rotateLL() {
  a1 := obj.Left
  a2 := obj.Right

  obj.Left  = a1.Left
  obj.Right = a1
  a1 .Left  = a1.Right
  a1 .Right = a2

  obj.Value, a1.Value = a1.Value, obj.Value

  obj.Right.Balance = 0
  obj      .Balance = 0
}

func (obj *AvlNode) rotateLR() {
  a1 := obj.Left
  a2 :=  a1.Right

  a1 .Right = a2 .Left
  a2 .Left  = a2 .Right
  a2 .Right = obj.Right
  obj.Right = a2

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

  obj.Right = a1.Right
  obj.Left  = a1
  a1 .Right = a1.Left
  a1 .Left  = a2

  obj.Value, a1.Value = a1.Value, obj.Value

  obj.Left.Balance = 0
  obj     .Balance = 0
}

func (obj *AvlNode) rotateRL() {
  a1 := obj.Right
  a2 :=  a1.Left

  a1 .Left  =  a2.Right
  a2 .Right =  a2.Left
  a2 .Left  = obj.Left
  obj.Left  = a2

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
