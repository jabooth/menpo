from menpo.base import MenpoMissingDependencyError

try:
    from .vlfeat import (dsift, fast_dsift, vector_128_dsift,
                         hellinger_vector_128_dsift)
except MenpoMissingDependencyError:
    pass

try:
    from .menpowidgets import features_selection_widget
except MenpoMissingDependencyError:
    pass

try:
    from .cython import hog, sparse_hog, lbp
except MenpoMissingDependencyError:
    pass

# Remove from scope
del MenpoMissingDependencyError
