from typing import TypeVar,Generic
import numpy.typing as npt
import numpy as np

class ObjectArray(npt.NDArray[np.object_], Generic[TypeVar('T')]):
    """A numpy array that holds objects of type T"""
    pass