from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import math

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim, normalize_image
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.cam_utils import camera_normalization, convert_pose_to_4x4, depth_projector
from ...geometry.camera_emb import get_plucker_embedding

# from .backbone.vggt.models.vggt import VGGT
from .backbone.vggt.heads.dpt_gs_head import DPTGSHead
# from .backbone.vggt.heads.pose_head import PoseHead
from .backbone.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from .backbone.vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3, unproject_depth_map_to_point_map_batch
from ...misc.intrinsics_utils import normalize_intrinsics, recover_intrinsics

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderSPFSplatV2LCfg:
    name: Literal["spfsplatv2-l"]
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    pretrained_weights: str = ""
    pose_free: bool = True
    pose_make_baseline_1: bool = True
    pose_make_relative: bool = True
    
    estimating_focal: bool = False
    estimating_pose: bool = True




def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderSPFSplatV2L(Encoder[EncoderSPFSplatV2LCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderSPFSplatV2LCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = 14
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity



        self.embed_dim = 1024
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        

    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

        elif 'dpt' in head_type:
            self.gaussian_param_head = DPTGSHead(dim_in=2 * self.embed_dim, output_dim=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")
        
   
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        target: Optional[dict] = None,
    ) :
        device = context["image"].device
        b, v_cxt, _, h, w = context["image"].shape


        if target is not None:
            v_tgt = target["image"].shape[1]
            context_target = {
                "image": torch.cat([context["image"], target["image"]], dim=1),
                "intrinsics": torch.cat([context["intrinsics"], target["intrinsics"]], dim=1)
            }
            # Encode the context and target images.
            aggregated_tokens_list, ps_idx = self.backbone(context_target, target_num_views=v_tgt)
        else:
            v_tgt = 0
            aggregated_tokens_list, ps_idx = self.backbone(context, target_num_views=0)
            
        if self.cfg.estimating_pose:
            camera_enc = self.backbone.model.camera_head(aggregated_tokens_list)[-1] # [b, v, 9]
            extri, _ = pose_encoding_to_extri_intri(camera_enc, context["image"].shape[-2:])
            if self.cfg.estimating_pose:
                pred_extrinsics = self.process_pose(extri, v_cxt)
               

        context_aggregated_tokens_list = []
        for aggregated_tokens in aggregated_tokens_list:
            context_aggregated_tokens_list.append(aggregated_tokens[:,:v_cxt].contiguous())

        # Predict Point Maps
        point_map, _ = self.backbone.model.point_head(context_aggregated_tokens_list, context["image"], ps_idx) # [b, v, h, w, 3]
        # print("point_map", point_map.shape)
        pts_all = rearrange(point_map, "b v h w xyz -> b v (h w) xyz")
        extrinsics = pred_extrinsics[:, :v_cxt] if self.cfg.estimating_pose else context["extrinsics"]
        depths_per_view = self.process_depth(extrinsics, rearrange(pts_all, "b v (h w) xyz -> b v h w xyz", h=h, w=w)) # depth for each cam, (b, v, h, w)

        

        # Predict gaussians 
        gs_map = self.gaussian_param_head(context_aggregated_tokens_list, context["image"], ps_idx) # [b, v, h, w, 83]
        gaussians = rearrange(gs_map, "b v h w c -> b v (h w) c") 
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces) # for cfg.num_surfaces
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1) 
        gaussian_parameters = gaussians[..., 1:]
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces


        # Convert the features and depths into Gaussians.
        gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(gaussian_parameters, "b v r srf c -> b v r srf () c"),
        )
            
        
        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = depths_per_view

            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            ) # (b, v, h, w, 1, 3)
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ) # (b, v, h, w, 1, 1)

           

        encoder_output = dict()
        encoder_output["gaussians"] = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.rotations,
                "b v r srf spp i  -> b (v r srf spp) i ",
            ),
            rearrange(
                gaussians.scales,
                "b v r srf spp i  -> b (v r srf spp) i ",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            )
        )

        

        if self.cfg.estimating_pose:
            encoder_output['extrinsics'] = dict()
            encoder_output['extrinsics']['c'] = pred_extrinsics[:,:v_cxt]
            if target is not None:
                encoder_output['extrinsics']['cwt'] = pred_extrinsics


        
        return encoder_output

    def process_pose(self, poses, context_views):
        
        b, v = poses.shape[:2]

        poses = closed_form_inverse_se3(rearrange(poses, "b v ... -> (b v) ...")) # world to cam -> cam to world 
        poses = rearrange(poses, "(b v) ... -> b v ...", b=b, v=v)


        if self.cfg.pose_make_baseline_1:
            a = poses[:, 0, :3, 3]  # [b, 3]
            b = poses[:, context_views - 1, :3, 3]  #  [b, 3]

            scale = (a - b).norm(dim=1, keepdim=True)  # [b, 1]

            poses[:, :, :3, 3] /= scale.unsqueeze(-1)

        if self.cfg.pose_make_relative:
            base_context_pose = poses[:,0] # [b, 4, 4]
            inv_base_context_pose = torch.inverse(base_context_pose)
            poses = inv_base_context_pose[:, None, :, :] @ poses # [b,1,4,4] @ [b,v,4,4]

        return poses      
    
    def process_depth(self, pose, pts3d):
        b, v, h, w, _ = pts3d.shape
        pts3d = rearrange(pts3d, "b v h w c -> (b v) (h w) c")
        pose = rearrange(pose, "b v ... -> (b v) ...")


        depths = depth_projector(pts3d, pose) # (bv, n, 1)
        depths = rearrange(depths, "(b v) (h w) 1 -> b v h w", b=b, v=v, h=h, w=w)
        return depths.contiguous()
    

    

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
