from typing import Any
import torch.nn as nn

from .backbone import Backbone
from .backbone_croco_multiview import AsymmetricCroCoMulti, BackboneCrocoCfg
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_croco import AsymmetricCroCo
from .backbone_masked_croco import AsymmetricMaskedCroCoMulti, BackboneMaskedCrocoMultiCfg
from .backbone_masked_vggt import BackboneMaskedVGGT, BackboneMaskedVGGTCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "croco": AsymmetricCroCo,
    "croco_multi": AsymmetricCroCoMulti,
    "masked_croco_multi": AsymmetricMaskedCroCoMulti,
    "masked_vggt": BackboneMaskedVGGT
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg | BackboneCrocoCfg | BackboneMaskedCrocoMultiCfg | BackboneMaskedVGGTCfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3) -> nn.Module:
    return BACKBONES[cfg.name](cfg, d_in)
