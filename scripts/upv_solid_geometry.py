"""Physical occupied boxes shared by open-edge and pad-removal diagnostics."""
import copy
from gas_void_geometry import cell_box


def physical_obstacles(objects,shells,removed=()):
    result=[]
    for ident,c in objects.items():
        if ident=='chamber-wall' or ident in removed:continue
        box=copy.deepcopy(c)
        if ident in shells:
            box['geometry']['dimensions_m']=[v+2*shells[ident]['thickness_m']for v in c['geometry']['dimensions_m']]
        result.append(cell_box(box))
    return result
