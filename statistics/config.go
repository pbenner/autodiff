/* Copyright (C) 2016 Philipp Benner
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

package statistics

/* -------------------------------------------------------------------------- */

import "fmt"
import "encoding/json"
import "io"
import "io/ioutil"
import "reflect"
import "os"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type ConfigDistribution struct {
	Name          string
	Parameters    interface{}
	Distributions []ConfigDistribution
}

func NewConfigDistribution(name string, parameters interface{}, distributions ...ConfigDistribution) ConfigDistribution {
	var p interface{}
	switch parameters := parameters.(type) {
	case []float64:
		p = parameters
	case Vector:
		s := make([]float64, parameters.Dim())
		for i := 0; i < len(s); i++ {
			s[i] = parameters.At(i).GetValue()
		}
		p = s
	default:
		p = parameters
	}
	return ConfigDistribution{name, p, distributions}
}

func (config *ConfigDistribution) ReadJson(reader io.Reader) error {
	b, err := ioutil.ReadAll(reader)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, config)
}

func (config *ConfigDistribution) ImportJson(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	return config.ReadJson(f)
}

func (config ConfigDistribution) WriteJson(writer io.Writer) error {
	b, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	if _, err := writer.Write(b); err != nil {
		return err
	}
	return nil
}

func (config ConfigDistribution) ExportJson(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return config.WriteJson(f)
}

/* -------------------------------------------------------------------------- */

func (config ConfigDistribution) getBool(a interface{}) (bool, bool) {
	switch reflect.TypeOf(a).Kind() {
	case reflect.Bool:
		return bool(reflect.ValueOf(a).Bool()), true
	}
	return false, false
}

func (config ConfigDistribution) getFloat(a interface{}) (float64, bool) {
	switch reflect.TypeOf(a).Kind() {
	case reflect.Float64:
		return reflect.ValueOf(a).Float(), true
	}
	return 0, false
}

func (config ConfigDistribution) getInt(a interface{}) (int, bool) {
	switch reflect.TypeOf(a).Kind() {
	case reflect.Float64:
		return int(reflect.ValueOf(a).Float()), true
	}
	return 0, false
}

func (config ConfigDistribution) getString(a interface{}) (string, bool) {
	switch reflect.TypeOf(a).Kind() {
	case reflect.String:
		return reflect.ValueOf(a).String(), true
	}
	return "", false
}

func (config ConfigDistribution) getFloats(a interface{}) ([]float64, bool) {
	if a == nil {
		return nil, true
	}
	switch reflect.TypeOf(a).Kind() {
	case reflect.Slice:
		s := reflect.ValueOf(a)
		p := make([]float64, s.Len())
		for i := 0; i < s.Len(); i++ {
			if v, ok := config.getFloat(s.Index(i).Elem().Interface()); !ok {
				return nil, false
			} else {
				p[i] = v
			}
		}
		return p, true
	}
	return nil, false
}

func (config ConfigDistribution) getInts(a interface{}) ([]int, bool) {
	if a == nil {
		return nil, true
	}
	switch reflect.TypeOf(a).Kind() {
	case reflect.Slice:
		s := reflect.ValueOf(a)
		p := make([]int, s.Len())
		for i := 0; i < s.Len(); i++ {
			if v, ok := config.getInt(s.Index(i).Elem().Interface()); !ok {
				return nil, false
			} else {
				p[i] = v
			}
		}
		return p, true
	}
	return nil, false
}

func (config ConfigDistribution) getStrings(a interface{}) ([]string, bool) {
	if a == nil {
		return nil, true
	}
	switch reflect.TypeOf(a).Kind() {
	case reflect.Slice:
		s := reflect.ValueOf(a)
		p := make([]string, s.Len())
		for i := 0; i < s.Len(); i++ {
			if v, ok := config.getString(s.Index(i).Elem().Interface()); !ok {
				return nil, false
			} else {
				p[i] = v
			}
		}
		return p, true
	}
	return nil, false
}

func (config ConfigDistribution) GetParametersAsFloats() ([]float64, bool) {
	return config.getFloats(config.Parameters)
}

func (config ConfigDistribution) GetParametersAsVector(t ScalarType) (Vector, bool) {
	if v, ok := config.getFloats(config.Parameters); !ok {
		return nil, false
	} else {
		return NewVector(t, v), true
	}
}

func (config ConfigDistribution) GetParametersAsMatrix(t ScalarType, n, m int) (Matrix, bool) {
	if v, ok := config.getFloats(config.Parameters); !ok {
		return nil, false
	} else {
		return NewMatrix(t, n, m, v), true
	}
}

func (config ConfigDistribution) GetNamedParameter(name string) (interface{}, bool) {
	switch reflect.TypeOf(config.Parameters).Kind() {
	case reflect.Map:
		s := reflect.ValueOf(config.Parameters)
		r := s.MapIndex(reflect.ValueOf(name))
		if r.IsValid() {
			return r.Interface(), true
		}
	}
	return 0, false
}

func (config ConfigDistribution) GetNamedParametersAsFloats(name string) ([]float64, bool) {
	if p, ok := config.GetNamedParameter(name); ok {
		return config.getFloats(p)
	}
	return nil, false
}

func (config ConfigDistribution) GetNamedParametersAsInts(name string) ([]int, bool) {
	if p, ok := config.GetNamedParameter(name); ok {
		return config.getInts(p)
	}
	return nil, false
}

func (config ConfigDistribution) GetNamedParametersAsStrings(name string) ([]string, bool) {
	if p, ok := config.GetNamedParameter(name); ok {
		return config.getStrings(p)
	}
	return nil, false
}

func (config ConfigDistribution) GetNamedParameterAsScalar(name string, t ScalarType) (Scalar, bool) {
	if v, ok := config.getFloat(config.Parameters); !ok {
		return nil, false
	} else {
		return NewScalar(t, v), true
	}
}

func (config ConfigDistribution) GetNamedParameterAsBool(name string) (bool, bool) {
	if p, ok := config.GetNamedParameter(name); ok {
		return config.getBool(p)
	}
	return false, false
}

func (config ConfigDistribution) GetNamedParameterAsFloat(name string) (float64, bool) {
	if p, ok := config.GetNamedParameter(name); ok {
		return config.getFloat(p)
	}
	return 0, false
}

func (config ConfigDistribution) GetNamedParameterAsInt(name string) (int, bool) {
	if p, ok := config.GetNamedParameter(name); ok {
		return config.getInt(p)
	}
	return 0, false
}

func (config ConfigDistribution) GetNamedParameterAsString(name string) (string, bool) {
	if p, ok := config.GetNamedParameter(name); ok {
		return config.getString(p)
	}
	return "", false
}

func (config ConfigDistribution) GetNamedParametersAsVector(name string, t ScalarType) (Vector, bool) {
	if v, ok := config.GetNamedParametersAsFloats(name); !ok {
		return nil, false
	} else {
		return NewVector(t, v), true
	}
}

func (config ConfigDistribution) GetNamedParametersAsMatrix(name string, t ScalarType, n, m int) (Matrix, bool) {
	if v, ok := config.GetNamedParametersAsFloats(name); !ok {
		return nil, false
	} else {
		return NewMatrix(t, n, m, v), true
	}
}

func (config ConfigDistribution) getNestedInts(a interface{}) (interface{}, bool) {
	if a == nil {
		return nil, true
	}
	switch reflect.TypeOf(a).Kind() {
	case reflect.Slice:
		s := reflect.ValueOf(a)
		p := make([]interface{}, s.Len())
		for i := 0; i < s.Len(); i++ {
			if v, ok := config.getInt(s.Index(i).Elem().Interface()); !ok {
				if v, ok := config.getNestedInts(s.Index(i).Elem().Interface()); !ok {
					return nil, false
				} else {
					p[i] = v
				}
			} else {
				p[i] = v
			}
		}
		return p, true
	}
	return nil, false
}

func (config ConfigDistribution) GetNamedParametersAsNestedInts(name string) (interface{}, bool) {
	if p, ok := config.GetNamedParameter(name); ok {
		return config.getNestedInts(p)
	}
	return 0, false
}

func (config ConfigDistribution) GetNamedParametersAsIntPairs(name string) ([][2]int, bool) {
	if r, ok := config.GetNamedParametersAsNestedInts(name); ok {
		if a, ok := r.([]interface{}); !ok {
			return nil, false
		} else {
			pairs := make([][2]int, len(a))
			for i, ai := range a {
				if b, ok := ai.([]interface{}); !ok {
					return nil, false
				} else {
					if len(b) != 2 {
						return nil, false
					}
					if v, ok := b[0].(int); !ok {
						return nil, false
					} else {
						pairs[i][0] = v
					}
					if v, ok := b[1].(int); !ok {
						return nil, false
					} else {
						pairs[i][1] = v
					}
				}
			}
			return pairs, true
		}
	}
	return nil, false
}

/* -------------------------------------------------------------------------- */

func ExportDistribution(filename string, distribution ConfigurableDistribution) error {
	return distribution.ExportConfig().ExportJson(filename)
}

func ImportDistribution(filename string, distribution ConfigurableDistribution, t ScalarType) error {
	config := ConfigDistribution{}

	if err := config.ImportJson(filename); err != nil {
		return err
	}
	if err := distribution.ImportConfig(config, t); err != nil {
		return err
	}
	return nil
}

/* -------------------------------------------------------------------------- */

func ImportScalarPdfConfig(config ConfigDistribution, t ScalarType) (ScalarPdf, error) {
	if distribution := NewScalarPdf(config.Name); distribution == nil {
		return nil, fmt.Errorf("unknown distribution: %s", config.Name)
	} else {
		if err := distribution.ImportConfig(config, t); err != nil {
			return nil, err
		}
		return distribution, nil
	}
}

func ImportScalarPdf(filename string, t ScalarType) (ScalarPdf, error) {
	config := ConfigDistribution{}
	if err := config.ImportJson(filename); err != nil {
		return nil, err
	}
	return ImportScalarPdfConfig(config, t)
}

/* -------------------------------------------------------------------------- */

func ImportVectorPdfConfig(config ConfigDistribution, t ScalarType) (VectorPdf, error) {
	if distribution := NewVectorPdf(config.Name); distribution == nil {
		return nil, fmt.Errorf("unknown distribution: %s", config.Name)
	} else {
		if err := distribution.ImportConfig(config, t); err != nil {
			return nil, err
		}
		return distribution, nil
	}
}

func ImportVectorPdf(filename string, t ScalarType) (VectorPdf, error) {
	config := ConfigDistribution{}
	if err := config.ImportJson(filename); err != nil {
		return nil, err
	}
	return ImportVectorPdfConfig(config, t)
}

/* -------------------------------------------------------------------------- */

func ImportMatrixPdfConfig(config ConfigDistribution, t ScalarType) (MatrixPdf, error) {
	if distribution := NewMatrixPdf(config.Name); distribution == nil {
		return nil, fmt.Errorf("unknown distribution: %s", config.Name)
	} else {
		if err := distribution.ImportConfig(config, t); err != nil {
			return nil, err
		}
		return distribution, nil
	}
}

func ImportMatrixPdf(filename string, t ScalarType) (MatrixPdf, error) {
	config := ConfigDistribution{}
	if err := config.ImportJson(filename); err != nil {
		return nil, err
	}
	return ImportMatrixPdfConfig(config, t)
}
