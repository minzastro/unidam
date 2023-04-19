import numpy as np
from scipy.optimize import curve_fit
from unidam.utils.mathematics import kl_divergence, wstatistics
from unidam.fitters import basic
from unidam.utils.extra_functions import unidam_extra_functions as uef
from scipy.stats import norm

def tgauss(dummy, x, mu, sigma, lower, upper):
    """
    Proxy for truncated Gaussian.
    """
    result = np.zeros_like(x)
    mask = (x >= lower) * (x <= upper)
    result[mask] = uef.trunc_normal(x[mask], mu, sigma, lower, upper)
    return result


class TGaussianFit(basic.PdfFitter):
    USE_TRF = True

    LETTER = 'T'

    FUNC = tgauss

    def _get_residual_sum_sq(self, solution):
        return np.sum(self._get_residual_values(solution) ** 2)

    def _get_residual_values(self, solution):
        return self.y - self.FUNC(self.x,
                                  solution[0][0],
                                  solution[0][1],
                                  self.x[solution[1]],
                                  self.x[solution[2]])

    def _get_function(self, solution):
        def fit_fun(x, mu, sigma):
            return uef.trunc_normal(x, mu, sigma,
                                    self.x[solution[1]],
                                    self.x[solution[2]])
        return fit_fun

    def _local_fit(self, solution):
        try:
            popt, pcov = curve_fit(self._get_function(solution),
                                   self.x, self.y,
                                   solution[0], method='trf',
                                   ftol=1e-4, bounds=self.bounds)
            if self.is_solution_ok(popt, pcov):
                return popt
            else:
                return self.init_params
        except ValueError:
            return self.init_params

    def _move_lower(self, solution):
        y_pred = norm.pdf(x=self.x, loc=solution[0][0], scale=solution[0][1])
        #import ipdb; ipdb.set_trace()
        lower = solution[1]
        while lower > 0:
            lower -= 1
            #print('-->', lower, self.y[lower], y_pred[lower])
            if np.abs(y_pred[lower]) < np.abs(self.y[lower] - y_pred[lower]):
                lower += 1
                break
        new_solution = [solution[0], lower - 1, solution[2], 0, False]
        #print(solution, new_solution)
        test = self._local_fit(new_solution)
        if test is not None:
            new_solution[0] = test
            residual = self._get_residual_sum_sq(new_solution)
            new_solution[3] = residual
            if residual < solution[3]:
                new_solution[4] = True
            else:
                new_solution[1] += 1
            if new_solution[1] == 0:
                new_solution[4] = False
        else:
            new_solution[1] += 1
        return new_solution

    def _move_upper(self, solution):
        new_solution = [solution[0], solution[1], solution[2] + 1, 0, False]
        test = self._local_fit(new_solution)
        if test is not None:
            new_solution[0] = test
            residual = self._get_residual_sum_sq(new_solution)
            new_solution[3] = residual
            if residual < solution[3]:
                new_solution[4] = True
            else:
                new_solution[2] -= 1
            if new_solution[2] >= len(self.x) - 1:
                new_solution[4] = False
        else:
            new_solution[2] -= 1
        return new_solution

    def _fit(self):
        modepos = np.argmax(self.y)
        w = np.where(self.y > self.y.max() * 0.2)[0]
        self.lower = w[0]
        self.upper = w[-1]
        if self.lower == self.upper:
            self.lower = max(self.lower - 1, 0)
            self.upper = min(self.upper + 1, len(self.x) - 1)
        solution = [wstatistics(self.x, self.y, 2),
                    self.lower, self.upper,
                    0, True]
        solution[0] = self._local_fit(solution)
        solution[3] = self._get_residual_sum_sq(solution)
        if modepos > 0:
            while solution[-1] and self.lower > 0:
                # Increase lower bound gradually,
                # re-fitting at each step, while residuals decrease.
                solution = self._move_lower(solution)
        else:
            solution[1] = 0
        solution[-1] = True
        if self.upper > modepos and self.upper < len(self.x) - 1:
            while solution[-1]:
                # Increase lower bound gradually,
                # re-fitting at each step, while residuals decrease.
                solution = self._move_upper(solution)
        best = solution[0]
        result = [best[0], best[1], self.x[solution[1]], self.x[solution[2]]]
        return [result, kl_divergence(self.x, self.FUNC, result, self.y)]

    def is_applicable(self):
        return len(self.x) > 2
