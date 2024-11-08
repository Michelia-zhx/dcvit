from typing import Type, Any, Callable, Union, List, Optional
from functools import partial

import torch
import torch.nn as nn
import torchvision

# from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
# from torchvision.models import Swin_B_16_Weights
import timm
from timm.models.swin_transformer import SwinTransformer
# from torchvision.models.vision_transformer import ConvStemConfig
# from torchvision.models._api import WeightsEnum

class Swin(SwinTransformer):
    
    def __init__(
            self,
            img_size=224, 
            patch_size=4, 
            in_chans=3, 
            num_classes=1000,
            embed_dim=96, 
            origin_depths = (2, 2, 18, 2),
            depths=(2, 2, 18, 2), 
            num_heads=(3, 6, 12, 24), 
            window_size=7, 
            mlp_ratio=4., 
            drop_rate=0., 
            attn_drop_rate=0., 
            drop_path_rate=0.1,
            weight_init=''
    ):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                        embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, 
                        mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
                        drop_path_rate=drop_path_rate, weight_init=weight_init)
        self.get_feat = 'None'
        # self.block_o_p = dict()
        # self.first_neighbour_block = 0
        self.freeze_setting = 0
        self.replaced_blocks = []
        self.GELU_array = None
        self.training = False
        self.attn_map = None
        self.origin_depths = origin_depths
        self.depths = depths
        self.depth = sum(depths)
        self.last_neighbour_block = self.depth - 1

    def forward(self, x: torch.Tensor):
        if self.get_feat == 'pre_GAP':
            f_before = self.forward_features(x)
            x = self.forward_head(f_before)
            return x, f_before
        elif self.get_feat == 'after_GAP':
            f_before = self.forward_features(x)
            f_after = self.forward_head(f_before, pre_logits=True)
            with torch.no_grad():
                x = self.forward_head(f_before)
            return x, f_after
        else:
            f_before = self.forward_features(x)
            x = self.forward_head(f_before)
            return x
    
    def get_blocks_to_drop(self):
        nums = len(self.blocks)
        blocks = [str(x) for x in range(nums)]
        return blocks

    def freeze_classifier(self):
        for param in self.head.parameters():
            param.requires_grad = False

def _swin(
    rm_blocks,
    patch_size,
    embed_dim,
    depths,
    num_heads,
    drop_path_rate,
    weight_init='',
    **kwargs: Any,
) -> Swin:
    img_size = kwargs.pop("image_size", 224)

    model = Swin(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )

    if weight_init:
        load_rm_block_state_dict(model, weight_init, rm_blocks)
    
    return model

def load_rm_block_state_dict(model, raw_state_dict, rm_blocks):
    rm_count = [0,0,0,0]
    has_count = set()
    # block_o_p = dict()
    # neighbour_blocks = set()
    state_dict = dict()
    raw_state_dict = {k.replace('module.',''):v for k,v in raw_state_dict.items()}
    for raw_key in raw_state_dict.keys():
        key_items = raw_key.split('.')
        if len(key_items) > 1 and key_items[0] == 'layers' and key_items[2] == 'blocks' and 'total' not in key_items[-1]:
            layer_key = key_items[1]
            block_key = key_items[3]
            block_idx = layer2block(layer_key, block_key)
            if block_idx in rm_blocks:
                # print(layer_key, block_key)
                if block_idx not in has_count:
                    has_count.add(block_idx)
                    rm_count[int(layer_key)] += 1
            else:
                new_block_key = str(int(block_key) - rm_count[int(layer_key)])
                # block_o_p[(layer_key, block_key)] = (layer_key, new_block_key)
                key_items[3] = new_block_key
                target_key = '.'.join(key_items)
                # assert target_key in state_dict, f'{raw_key} -> {target_key}'
                state_dict[target_key] = raw_state_dict[raw_key]
        elif 'total' not in key_items[-1]:
            # assert raw_key in state_dict
            state_dict[raw_key] = raw_state_dict[raw_key]
    # print(f'block_o_p: {block_o_p}')
    # print(f'neighbour_blocks: {neighbour_blocks}')
    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    # model.block_o_p = block_o_p
    if len(rm_blocks) != 0:
        # model.neighbour_blocks = neighbour_blocks
        # model.first_neighbour_block = int(min(neighbour_blocks))
        model.last_neighbour_block = block2layer(str(int(max(rm_blocks))+1))
        # print(f'first unfrozen block: {model.first_neighbour_block}, last unfrozen block: {model.last_neighbour_block}')


def rm_block_from_origin(origin_model, rm_blocks, LOG, args, pretrained=True, num_classes=1000, teacher=True, **kwargs: Any) -> Swin:
    weights = origin_model.state_dict()
    if args.model == 'swin_base_patch4_window7_224':
        patch_size=4
        embed_dim=128
        depths=(2, 2, 18, 2)
        num_heads=(4, 8, 16, 32)
    else:
        raise ValueError("{} not found".format(args.model))
    if rm_blocks != []:
        depths = origin_model.depths
    # print(rm_blocks, len(rm_blocks))
    # assert(0)
    depths = list(depths)
    for block in rm_blocks:
        (layer_id, block_id) = block2layer(block)
        depths[layer_id] -= 1
        assert(depths[layer_id] >= 0)
    depths = tuple(depths)
    if len(rm_blocks) > 0:
        LOG.info(f'=> build Swin_b_16 with layers {depths}. Remove {rm_blocks}')

    model = _swin(
        rm_blocks=rm_blocks,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        drop_path_rate=args.drop_path,
        weight_init=weights,
        **kwargs,
    )

    return model

def block2layer(block, depths=(2,2,18,2)):
    layer_id = 0
    for layer in range(len(depths)):
        if int(block) >= sum(depths[:layer+1]):
            layer_id += 1
        block_id = int(block) - sum(depths[:layer_id])
    return (layer_id, block_id)

def layer2block(layer_id, block_id, depths=(2,2,18,2)):
    block = sum(depths[:int(layer_id)]) + int(block_id)
    return str(block)

def get_blocks_to_drop(model):
    blocks = [str(x) for x in range(model.depth)]
    return blocks
