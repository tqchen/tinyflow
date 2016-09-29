"""Tinyflow trial."""
from __future__ import absolute_import as _abs
from . import _base
from nnvm.symbol import *
from . import nn
from . import train

from ._base import *
from ._ops import *

from ._session import Session

from ._util import infer_variable_shapes
