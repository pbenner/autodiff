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

//import "fmt"
import "testing"

/* -------------------------------------------------------------------------- */

func TestAvlTree1(test *testing.T) {

  tree := AvlTree{}
  tree.Insert(10)
  tree.Insert(20)

  v := []int{10, 20}
  b := []int{ 1,  0}

  for i, it := 0, tree.Iterator(); it.Ok(); it.Next() {
    switch {
    case i >= len(v):
      test.Error("test failed")
    case it.Get().Value != v[i]:
      test.Error("test failed")
    case it.Get().Balance != b[i]:
      test.Error("test failed")
    }
    i++
  }
}

func TestAvlTree2(test *testing.T) {

  tree := AvlTree{}
  tree.Insert(10)
  tree.Insert(9)

  v := []int{ 9, 10}
  b := []int{ 0, -1}

  for i, it := 0, tree.Iterator(); it.Ok(); it.Next() {
    switch {
    case i >= len(v):
      test.Error("test failed")
    case it.Get().Value != v[i]:
      test.Error("test failed")
    case it.Get().Balance != b[i]:
      test.Error("test failed")
    }
    i++
  }
}

func TestAvlTree3(test *testing.T) {

  tree := AvlTree{}
  tree.Insert(10)
  tree.Insert(20)
  tree.Insert(9)
  tree.Insert(12)

  v := []int{ 9, 10, 12, 20}
  b := []int{ 0,  1,  0, -1}

  for i, it := 0, tree.Iterator(); it.Ok(); it.Next() {
    switch {
    case i >= len(v):
      test.Error("test failed")
    case it.Get().Value != v[i]:
      test.Error("test failed")
    case it.Get().Balance != b[i]:
      test.Error("test failed")
    }
    i++
  }
}

func TestAvlTree4(test *testing.T) {

  tree := AvlTree{}
  tree.Insert(10)
  tree.Insert(20)
  tree.Insert(12)

  v := []int{10, 12, 20}
  b := []int{ 0,  0,  0}

  for i, it := 0, tree.Iterator(); it.Ok(); it.Next() {
    switch {
    case i >= len(v):
      test.Error("test failed")
    case it.Get().Value != v[i]:
      test.Error("test failed")
    case it.Get().Balance != b[i]:
      test.Error("test failed")
    }
    i++
  }
}

func TestAvlTree5(test *testing.T) {

  tree := AvlTree{}
  tree.Insert(1)
  tree.Insert(2)
  tree.Insert(3)
  tree.Insert(4)
  tree.Insert(5)
  tree.Insert(6)
  tree.Insert(7)

  v := []int{1, 2, 3, 4, 5, 6, 7}
  b := []int{0, 0, 0, 0, 0, 0, 0}

  for i, it := 0, tree.Iterator(); it.Ok(); it.Next() {
    switch {
    case i >= len(v):
      test.Error("test failed")
    case it.Get().Value != v[i]:
      test.Error("test failed")
    case it.Get().Balance != b[i]:
      test.Error("test failed")
    }
    i++
  }
}

func TestAvlTree6(test *testing.T) {

  tree := AvlTree{}
  tree.Insert(50)
  tree.Insert(25)
  tree.Insert(10)
  tree.Insert( 5)
  tree.Insert( 7)
  tree.Insert( 3)
  tree.Insert(30)
  tree.Insert(20)
  tree.Insert( 8)
  tree.Insert(15)

  v := []int{3,  5,  7, 8, 10, 15, 20, 25, 30, 50}
  b := []int{0, -1, -1, 0,  0,  0, -1,  0,  0, -1}

  for i, it := 0, tree.Iterator(); it.Ok(); it.Next() {
    switch {
    case i >= len(v):
      test.Error("test failed")
    case it.Get().Value != v[i]:
      test.Error("test failed")
    case it.Get().Balance != b[i]:
      test.Error("test failed")
    }
    i++
  }
}

func TestAvlTree7(test *testing.T) {

  tree := AvlTree{}
  tree.Insert(50)
  tree.Insert(25)
  tree.Insert(10)
  tree.Insert( 5)
  tree.Insert( 7)
  tree.Insert( 3)
  tree.Insert(30)
  tree.Insert(20)
  tree.Insert( 8)
  tree.Insert(15)

  tree.Delete(25)

  v := []int{3,  5,  7, 8, 10, 15, 20, 30, 50}
  b := []int{0, -1, -1, 0,  0,  0,  1,  0, -1}

  for i, it := 0, tree.Iterator(); it.Ok(); it.Next() {
    switch {
    case i >= len(v):
      test.Error("test failed")
    case it.Get().Value != v[i]:
      test.Error("test failed")
    case it.Get().Balance != b[i]:
      test.Error("test failed")
    }
    i++
  }
}