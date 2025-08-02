from typing import TYPE_CHECKING, TypeAlias, Any
from ._dependencies import np

_AnyShape: TypeAlias = tuple[Any, ...]
if TYPE_CHECKING:
    from .handlers import clusterClass
    from .dataAnalysis import dataAnalysis

    clusterArray: TypeAlias = list[clusterClass]
else:
    clusterClass: TypeAlias = object
    clusterArray: TypeAlias = list[clusterClass]
    dataAnalysis: TypeAlias = object
