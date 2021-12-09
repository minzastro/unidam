import numpy as np
from scipy.stats.stats import mode
from unidam.fitters import basic
from unidam.utils.mathematics import kl_divergence


def trunc_line(xin, val, val2, lower, upper):
    """
    Trapezoidal distribution:
    f(x) = val * x + val2, if lower <= x <= upper,
           0, otherwise.
    """
    result = np.zeros_like(xin)
    mask = (xin >= lower) * (xin <= upper)
    result[mask] = np.polyval([val, val2], xin[mask])
    result[result < 0.] = 0.
    return result


class LinearFit(basic.PdfFitter):
    LETTER = 'F'

    FUNC = trunc_line

    def _get_residual(self, solution):
        return np.sum((self.y[solution[1]:solution[2]] -
                      np.polyval(solution[0],
                                 self.x[solution[1]:solution[2]])) ** 2) + \
               np.sum(self.y[:solution[1]] ** 2) + \
               np.sum(self.y[solution[2]:] ** 2)

    def _move_lower(self, solution):
        test = np.polyfit(self.x[solution[1] - 1:solution[2]],
                          self.y[solution[1] - 1:solution[2]], 1)
        new_solution = [test, solution[1] - 1, solution[2], 0, False]
        residual = self._get_residual(new_solution)
        new_solution[3] = residual
        if residual < solution[3]:
            new_solution[4] = True
        else:
            new_solution[1] += 1
        if new_solution[1] == 0:
            new_solution[4] = False
        return new_solution

    def _move_upper(self, solution):
        test = np.polyfit(self.x[solution[1]:solution[2] + 1],
                          self.y[solution[1]:solution[2] + 1], 1)
        new_solution = [test, solution[1], solution[2] + 1, 0, False]
        residual = self._get_residual(solution)
        new_solution[3] = residual
        if residual < solution[3]:
            new_solution[4] = True
        else:
            new_solution[2] -= 1
        if new_solution[2] >= len(self.x) - 1:
            new_solution[4] = False
        return new_solution

    def _fit(self):
        self.test = np.polyfit(self.x, self.y, 1)
        modepos = np.argmax(self.y)
        w = np.where(self.y > self.y.max() * 0.2)[0]
        self.lower = w[0]
        self.upper = w[-1]
        solution = [self.test, self.lower, self.upper,
                    0, True]
        solution[3] = self._get_residual(solution)
        self.new_test = self.test
        if modepos > 0:
            while solution[-1] and solution[1] > 0:
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
        return [result, kl_divergence(self.x, trunc_line, result, self.y)]
