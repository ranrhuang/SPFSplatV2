from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange
from torch import nn

from .croco.blocks import DecoderBlock
from .croco.croco import CroCoNet
from .croco.misc import fill_default_args, freeze_all_params, transpose_to_landscape, is_symmetrized, interleave, \
    make_batch_symmetric
from .croco.patch_embed import get_patch_embed
from .backbone import Backbone
from ....geometry.camera_emb import get_intrinsic_embedding, get_intrinsic_positional_embedding

inf = float('inf')


croco_params = {
    'ViTLarge_BaseDecoder': {
        'enc_depth': 24,
        'dec_depth': 12,
        'enc_embed_dim': 1024,
        'dec_embed_dim': 768,
        'enc_num_heads': 16,
        'dec_num_heads': 12,
        'pos_embed': 'RoPE100',
        'img_size': (512, 512),
    },
}

default_dust3r_params = {
    'enc_depth': 24,
    'dec_depth': 12,
    'enc_embed_dim': 1024,
    'dec_embed_dim': 768,
    'enc_num_heads': 16,
    'dec_num_heads': 12,
    'pos_embed': 'RoPE100',
    'patch_embed_cls': 'PatchEmbedDust3R',
    'img_size': (512, 512),
    'head_type': 'dpt',
    'output_mode': 'pts3d',
    'depth_mode': ('exp', -inf, inf),
    'conf_mode': ('exp', 1, inf)
}


@dataclass
class BackboneMaskedCrocoMultiCfg:
    name: Literal["masked_croco_multi"]
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token", "none"] = 'token'  # linear or dpt
    pose_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    pose_embed_type: Literal['learnable_token'] = 'learnable_token'

class AsymmetricMaskedCroCoMulti(CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneMaskedCrocoMultiCfg, d_in: int) -> None:
        self.cfg = cfg
        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type

        self.pose_embed_loc = cfg.pose_embed_loc
        self.pose_embed_type = cfg.pose_embed_type

        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_encoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
        elif self.intrinsics_embed_loc == 'decoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_decoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3

        self.patch_embed_cls = cfg.patch_embed_cls
        self.croco_args = fill_default_args(croco_params[cfg.model], CroCoNet.__init__)

        super().__init__(**croco_params[cfg.model])

        if cfg.asymmetry_decoder:
            self.dec_blocks2 = deepcopy(self.dec_blocks)  # This is used in DUSt3R and MASt3R

        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'pe_and_token':
            self.intrinsic_encoder = nn.Linear(9, 1024)

        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'ray_token':
            self.intrinsic_encoder = nn.Linear(768, 1024)

        if self.intrinsics_embed_loc != 'none' and self.intrinsics_embed_type == 'learnable_token':
            self.intrinsics_token = nn.Parameter(torch.randn(1, 1, 1, 1024))

        if self.pose_embed_loc != 'none' and self.pose_embed_type == 'learnable_token':
            self.pose_token = nn.Parameter(torch.randn(1, 1, 1, 1024))


        # self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        # self.set_freeze(freeze)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans)


    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ['none', 'mask', 'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_decoder':  [self.mask_token, self.patch_embed, self.enc_blocks, self.enc_norm, self.decoder_embed, self.dec_blocks, self.dec_blocks2, self.dec_norm],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def _encode_image(self, image, true_shape, intrinsics_embed=None, intrinsics_pos_embed=None, pose_embedding=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape) # (bv, 256, 1024), (bv, 256, 2)
        
        P = x.shape[1]
        if intrinsics_pos_embed is not None:
            x = x + intrinsics_pos_embed

        if intrinsics_embed is not None:

            if self.intrinsics_embed_type == 'linear':
                x = x + intrinsics_embed
            elif 'token' in self.intrinsics_embed_type:
                x = torch.cat((x, intrinsics_embed), dim=1)
                add_pos = pos[:, 0:1, :].clone()
                add_pos[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
                pos = torch.cat((pos, add_pos), dim=1)
          

        if pose_embedding is not None:
            x = torch.cat((x, pose_embedding), dim=1)
            add_pos = pos[:, 0:1, :].clone()
            add_pos[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
            pos = torch.cat((pos, add_pos), dim=1)

        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, P 

    def _decoder(self, feat, pos, intrinsics_embed=None,  pose_embedding=None, extra_embed=None, num_target=0):
        

        if intrinsics_embed is not None:
            if self.intrinsics_embed_type == 'linear':
                feat = feat + intrinsics_embed
            elif 'token' in self.intrinsics_embed_type:
                # print("pos", pos[0])
                feat = torch.cat((feat, intrinsics_embed), dim=2)
                add_pos = pos[:, :, 0:1, :].clone()
                add_pos[:, :, :, 0] += (pos[:, :, -1, 0].unsqueeze(-1) + 1)
                pos = torch.cat((pos, add_pos), dim=2)
          

        if pose_embedding is not None:
            feat = torch.cat((feat, pose_embedding), dim=2)
            add_pos = pos[:, :, 0:1, :].clone()
            add_pos[:, :, :, 0] += (pos[:, :, -1, 0].unsqueeze(-1) + 1)
            pos = torch.cat((pos, add_pos), dim=2)


        b, v, l, c = feat.shape

        final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v)
        final_output.append(f)

        def generate_ctx_views(x):
            b, v, l, c = x.shape
            ctx_views = x.unsqueeze(1).expand(b, v, v, l, c)
            mask = torch.arange(v).unsqueeze(0) != torch.arange(v).unsqueeze(1)
            ctx_views = ctx_views[:, mask].reshape(b, v, v - 1, l, c)  # B, V, V-1, L, C
            # ctx_views = ctx_views.flatten(2, 3)  # B, V, (V-1)*L, C
            return ctx_views.contiguous()

        def generate_masks(v, num_target):
            # Step 1: Boolean attention mask as before
            mask = torch.ones((v, v), dtype=torch.bool)
            mask.fill_diagonal_(False)

            num_context = v - num_target
            context_indices = list(range(num_context))
            target_indices = list(range(num_context, v))

            for i in context_indices:
                mask[i, target_indices] = False  # context cannot attend to targets

            # Step 2: Remove self-attention and stack into shape [v, v-1]
            bool_mask = []
            for i in range(v):
                row = torch.cat([mask[i, :i], mask[i, i+1:]])
                bool_mask.append(row)
            bool_mask = torch.stack(bool_mask)  # shape [v, v-1]

            # Step 3: Convert to additive mask: True -> 0, False -> -inf
            mask = torch.where(bool_mask, torch.tensor(0.0), float("-inf")) # [v, v-1]
            return mask  
        
        pos_ctx = generate_ctx_views(pos) # [b, v, v-1, l, 2]: [1, 3, 2, 258, 2]

        ## Two implementations are both OK.
        ## ************ mask v1 ************
        # pos_ctx = pos_ctx.flatten(2, 3)
        # masks = generate_masks(v, num_target) # [v, v-1]: [3, 2]
        # # print("masks", masks.shape, masks)
        # masks = masks.repeat_interleave(l, dim=1) # [v, (v-1)*l]
        # masks = masks.unsqueeze(1).repeat(1, l, 1) # [v, l, (v-1)*l]
        # masks = masks.unsqueeze(0).repeat(b, 1, 1, 1).to(feat.device) # [b, v, l, (v-1)*l]
        
        # for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
        #     feat_current = final_output[-1] # [b, v, l, c]
        #     feat_current_ctx = generate_ctx_views(feat_current) # [b, v, (v-1), l, c]
        #     feat_current_ctx = feat_current_ctx.flatten(2, 3)  # [b, v, (v-1)*l, c]
        #     # img1 side
        #     f1, _ = blk1(feat_current[:, 0].contiguous(), feat_current_ctx[:, 0].contiguous(), pos[:, 0].contiguous(), pos_ctx[:, 0].contiguous(), masks[:,0])
        #     f1 = f1.unsqueeze(1) # [b, 1,l, c]

        #     # img2 side
        #     f2, _ = blk2(rearrange(feat_current[:, 1:], "b v l c -> (b v) l c"),
        #                  rearrange(feat_current_ctx[:, 1:], "b v l c -> (b v) l c"),
        #                  rearrange(pos[:, 1:].contiguous(), "b v l c -> (b v) l c"),
        #                  rearrange(pos_ctx[:, 1:].contiguous(), "b v l c -> (b v) l c"),
        #                  rearrange(masks[:,1:], "b v l l2 -> (b v) l l2"))
        #     f2 = rearrange(f2, "(b v) l c -> b v l c", b=b, v=v-1) # [b, v-1, l, c]
        #     # store the result
        #     final_output.append(torch.cat((f1, f2), dim=1))

        ## ************ mask v2, more efficient ************
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            feat_current = final_output[-1] # [b, v, l, c]
            feat_current_ctx = generate_ctx_views(feat_current) # [b, v, (v-1), l, c]
            # img1 side
            f1, _ = blk1(feat_current[:, 0].contiguous(), feat_current_ctx[:, 0, :v-1-num_target].flatten(1,2).contiguous(), pos[:, 0].contiguous(), pos_ctx[:, 0, :v-1-num_target].flatten(1,2).contiguous())
            f1 = f1.unsqueeze(1)
            # img2 side
            f2, _ = blk2(rearrange(feat_current[:, 1:v-num_target], "b v l c -> (b v) l c"),
                         rearrange(feat_current_ctx[:, 1:v-num_target, :v-1-num_target].flatten(2,3), "b v l c -> (b v) l c"),
                         rearrange(pos[:, 1:v-num_target].contiguous(), "b v l c -> (b v) l c"),
                         rearrange(pos_ctx[:, 1:v-num_target, :v-1-num_target].flatten(2,3).contiguous(), "b v l c -> (b v) l c")
                         )  
            f2 = rearrange(f2, "(b v) l c -> b v l c", b=b, v=v-1-num_target)

            if num_target > 0:
                f2_tgt, _ = blk2(rearrange(feat_current[:, v-num_target:], "b v l c -> (b v) l c"),
                            rearrange(feat_current_ctx[:, v-num_target:].flatten(2,3), "b v l c -> (b v) l c"),
                            rearrange(pos[:, v-num_target:].contiguous(), "b v l c -> (b v) l c"),
                            rearrange(pos_ctx[:, v-num_target:].flatten(2,3).contiguous(), "b v l c -> (b v) l c")
                            )  
                f2_tgt = rearrange(f2_tgt, "(b v) l c -> b v l c", b=b, v=num_target)
                f2 = torch.cat([f2, f2_tgt], dim=1)

            
            # store the result
            final_output.append(torch.cat((f1, f2), dim=1))
        #######################################################

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        last_feat = rearrange(final_output[-1], "b v l c -> (b v) l c")
        last_feat = self.dec_norm(last_feat)
        final_output[-1] = rearrange(last_feat, "(b v) l c -> b v l c", b=b, v=v)
        return final_output

    def forward(self,
                context: dict,
                target_num_views: int = 0,
                symmetrize_batch=False,
                return_views=False,
                ):
        b, v, _, h, w = context["image"].shape
            
        images_all = context["image"]


        #****************** encoder *******************************
        # camera embedding in the encoder
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            intrinsic_embedding = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree) # (b, v, 25, h w)
            images_all = torch.cat((images_all, intrinsic_embedding), dim=2) # (b, v, 28, h w)

        
        intrinsic_embedding_all = None
        if self.intrinsics_embed_loc == 'encoder' and (self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'linear'):
            intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2))
            intrinsic_embedding_all = rearrange(intrinsic_embedding, "b v c -> (b v) c").unsqueeze(1)


        pose_embedding_all = None
        if self.pose_embed_loc == 'encoder' and self.pose_embed_type == 'learnable_token':
            pose_embedding = self.pose_token.expand(b, v, *self.pose_token.shape[2:]) 
            pose_embedding_all = rearrange(pose_embedding, "b v ... -> (b v) ...")


        # step 1: encoder input images
        images_all = rearrange(images_all, "b v c h w -> (b v) c h w")
        shape_all = torch.tensor(images_all.shape[-2:])[None].repeat(b*v, 1)


        feat, pos, P = self._encode_image(images_all, shape_all, intrinsic_embedding_all, pose_embedding=pose_embedding_all) # (bv, n_tokens, 1024)
        # print("encoder", feat.shape, pos.shape)
        #****************** decoder *******************************
        
        intrinsic_embedding = None
        if self.intrinsics_embed_loc == 'decoder' and (self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'linear'):
            intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2)).unsqueeze(2)
        
        pose_embedding = None
        if self.pose_embed_loc == 'decoder' and self.pose_embed_type == 'learnable_token':
            pose_embedding = self.pose_token.expand(b, v, *self.pose_token.shape[2:]) 
        

        feat = rearrange(feat, "(b v) l c -> b v l c", b=b, v=v)
        pos = rearrange(pos, "(b v) l c -> b v l c", b=b, v=v)

        # print("feat1", feat.shape)
        dec_feat = self._decoder(feat, pos.contiguous(), intrinsics_embed=intrinsic_embedding, pose_embedding=pose_embedding, num_target=target_num_views)

        pose_feat = None
        if self.pose_embed_loc != 'none' and self.pose_embed_type == 'learnable_token':
            pose_feat = []

        dec_feat = list(dec_feat)
        for i in range(len(dec_feat)):
            # print("dec_feat", dec_feat[i].shape)

            if self.pose_embed_loc != 'none' and self.pose_embed_type == 'learnable_token':
                pose_feat.append(dec_feat[i][:, :, -1:])

            dec_feat[i] = dec_feat[i][:, :, :P]

            



        
        shape = rearrange(shape_all, "(b v) c -> b v c", b=b, v=v)
        images = rearrange(images_all, "(b v) c h w -> b v c h w", b=b, v=v)


        out = dict()
        out['dec_feat'] = dec_feat
        out['shape'] = shape
        out['images'] = images
        
        if pose_feat is not None:
            out['pose_feat'] = pose_feat

        return out

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024
