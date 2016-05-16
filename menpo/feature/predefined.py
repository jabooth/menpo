from functools import partial
from .features import igo

double_igo = partial(igo, double_angles=True)
double_igo.__name__ = 'double_igo'
double_igo.__doc__ = igo.__doc__
