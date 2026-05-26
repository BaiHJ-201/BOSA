from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .TRIBE import TRIBE
from .tent import TENT
from .bn import BN
from .cotta import CoTTA
from .bosa import BOSA
from .ecotta import EcoTTA
from .das import DAS
from .palm import PALM
from typing import Type  # Import Type for type annotations.

def build_adapter(cfg) -> Type[BaseAdapter]: 
    if cfg.ADAPTER.NAME == "rotta":
        return RoTTA
    elif cfg.ADAPTER.NAME == "tribe":
        return TRIBE
    elif cfg.ADAPTER.NAME == "tent":
        return TENT
    elif cfg.ADAPTER.NAME == "bn":
        return BN
    elif cfg.ADAPTER.NAME == "cotta":
        return CoTTA
    elif cfg.ADAPTER.NAME == "bosa":
        return BOSA
    elif cfg.ADAPTER.NAME == "ecotta":
        return EcoTTA
    elif cfg.ADAPTER.NAME == "das":
        return DAS
    elif cfg.ADAPTER.NAME == "palm":
        return PALM
    else:
        raise NotImplementedError("Implement your own adapter")
