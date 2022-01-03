import numpy as np
from unidam.fitters import basic
from unidam.utils.extra_functions import unidam_extra_functions as uef


def t_student(dummy, x, mu, sigma, degrees_of_freedom):
    result = uef.student_pdf(x, mu, np.abs(sigma), np.abs(degrees_of_freedom))
    return result / (result.sum() * (x[1] - x[0]))


class StudentsFit(basic.PdfFitter):
    LETTER = 'P'

    FUNC = t_student

    USE_TRF = True

    def is_solution_ok(self, popt, pcov):
        if np.any(np.isinf(np.diag(pcov))):
            return False
        if np.any(np.sqrt(np.abs(np.diag(pcov)))[:-1] >
                  10. * np.abs(popt)[:-1]):
            return False
        return True

    def __init__(self, x, y):
        super(StudentsFit, self).__init__(x, y)
        nu0 = np.min([np.power(self.init_params[-1], -0.7), 1])
        self.init_params.append(nu0)

    def is_applicable(self):
        return len(self.x) > 2