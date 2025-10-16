from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .TRIBE import TRIBE
from .lame import LAME
from .tent import TENT
from .pl import PL
from .bn import BN
from .note import NOTE
from .ttac import TTAC
from .eata import EATA
from .cotta import CoTTA
from .petal import PETALFim
from .datta import DATTA
from .RT import RT
from .ecotta import EcoTTA
from typing import Type  # 导入Type用于类型注解
# 确保datta模块存在且拼写正确

def build_adapter(cfg) -> Type[BaseAdapter]: 
    if cfg.ADAPTER.NAME == "rotta":
        return RoTTA
    elif cfg.ADAPTER.NAME == "tribe":
        return TRIBE
    elif cfg.ADAPTER.NAME == "lame":
        return LAME
    elif cfg.ADAPTER.NAME == "tent":
        return TENT
    elif cfg.ADAPTER.NAME == "pl":
        return PL
    elif cfg.ADAPTER.NAME == "bn":
        return BN
    elif cfg.ADAPTER.NAME == "note":
        return NOTE
    elif cfg.ADAPTER.NAME == "ttac":
        return TTAC
    elif cfg.ADAPTER.NAME == "eata":
        return EATA
    elif cfg.ADAPTER.NAME == "cotta":
        return CoTTA
    elif cfg.ADAPTER.NAME == "petal":
        return PETALFim
    elif cfg.ADAPTER.NAME == "datta":
        return DATTA
    elif cfg.ADAPTER.NAME == "datta":
        return DATTA
    elif cfg.ADAPTER.NAME == "rt":
        return RT
    elif cfg.ADAPTER.NAME == "EcoTTA":
        return EcoTTA
    else:
        raise NotImplementedError("Implement your own adapter")

