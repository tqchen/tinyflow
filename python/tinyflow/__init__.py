"""Tinyflow trial."""
from __future__ import absolute_import
from . import _base
from nnvm.symbol import *
from . import nn
from . import train

from ._base import placeholder, Variable, float32

from ._session import Session
