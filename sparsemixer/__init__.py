# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__author__ = "Liyuan Liu"

__maintainer__ = "Liyuan Liu"
__email__ = "llychinalz@gmail.com"

from .sparsemixer import SparseMixer, SparseMixerFirst, SparseMixerInf
from .switchgate import SwitchGate

router_map = {
    'sparsemixer': SparseMixer, 
    'switchgate': SwitchGate,
    'sparsemixer-1st': SparseMixerFirst, 
    'sparsemixer-inf': SparseMixerInf, 
}

def get_router(name):
    name=name.lower()
    assert name in router_map, f'Currently only supports SparseMixer (-1st/inf) and SwitchGate. {name} is not supported!'
    
    return router_map[name]
