from typing import TYPE_CHECKING, TypeAlias, Any
from ._dependencies import np

_AnyShape: TypeAlias = tuple[Any, ...]
if TYPE_CHECKING:
    from .handlers import clusterClass
    from .dataAnalysis import dataAnalysis

    clusterArray: TypeAlias = np.ndarray[_AnyShape, clusterClass]
else:
    clusterClass: TypeAlias = object
    clusterArray: TypeAlias = np.ndarray[_AnyShape, clusterClass]
    dataAnalysis: TypeAlias = object
