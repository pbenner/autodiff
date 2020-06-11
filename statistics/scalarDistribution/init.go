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

package scalarDistribution

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

func init() {
	ScalarPdfRegistry["scalar:beta distribution"] = new(BetaDistribution)
	ScalarPdfRegistry["scalar:binomial distribution"] = new(BinomialDistribution)
	ScalarPdfRegistry["scalar:categorical distribution"] = new(CategoricalDistribution)
	ScalarPdfRegistry["scalar:cauchy distribution"] = new(CauchyDistribution)
	ScalarPdfRegistry["scalar:delta distribution"] = new(DeltaDistribution)
	ScalarPdfRegistry["scalar:exponential distribution"] = new(ExponentialDistribution)
	ScalarPdfRegistry["scalar:gamma distribution"] = new(GammaDistribution)
	ScalarPdfRegistry["scalar:generalized gamma distribution"] = new(GeneralizedGammaDistribution)
	ScalarPdfRegistry["scalar:geometric distribution"] = new(GeometricDistribution)
	ScalarPdfRegistry["scalar:gev distribution"] = new(GevDistribution)
	ScalarPdfRegistry["scalar:mixture distribution"] = new(Mixture)
	ScalarPdfRegistry["scalar:laplace distribution"] = new(LaplaceDistribution)
	ScalarPdfRegistry["scalar:negative binomial distribution"] = new(NegativeBinomialDistribution)
	ScalarPdfRegistry["scalar:normal distribution"] = new(NormalDistribution)
	ScalarPdfRegistry["scalar:pareto distribution"] = new(ParetoDistribution)
	ScalarPdfRegistry["scalar:generalized pareto distribution"] = new(GParetoDistribution)
	ScalarPdfRegistry["scalar:poisson distribution"] = new(PoissonDistribution)
	ScalarPdfRegistry["scalar:power law distribution"] = new(PowerLawDistribution)
	ScalarPdfRegistry["scalar:pdf log transform"] = new(PdfLogTransform)
	ScalarPdfRegistry["scalar:pdf translation"] = new(PdfTranslation)
}
