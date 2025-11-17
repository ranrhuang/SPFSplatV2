from typing import Optional

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .encoder_spfsplat import EncoderSPFSplatCfg, EncoderSPFSplat
from .encoder_spfsplatv2 import EncoderSPFSplatV2Cfg, EncoderSPFSplatV2
from .encoder_spfsplatv2l import EncoderSPFSplatV2LCfg, EncoderSPFSplatV2L

ENCODERS = {
    "spfsplat": (EncoderSPFSplat, None),
    "spfsplatv2": (EncoderSPFSplatV2, None),
    "spfsplatv2-l": (EncoderSPFSplatV2L, None),
}

EncoderCfg = EncoderSPFSplatCfg | EncoderSPFSplatV2Cfg | EncoderSPFSplatV2LCfg 

def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
