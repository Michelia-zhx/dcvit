from typing import Type, Any, Callable, Union, List, Optional
from functools import partial

import torch
import torch.nn as nn
import torchvision

# from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
# from torchvision.models import DeiT_B_16_Weights
import timm
from timm.models.vision_transformer import VisionTransformer
# from torchvision.models.vision_transformer import ConvStemConfig
# from torchvision.models._api import WeightsEnum

class DeiT(VisionTransformer):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""
    
    def __init__(
            self,
            img_size,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            in_chans=3,
            num_classes=1000,
            drop_path_rate=0.,
            weight_init='',
            update_ln=True
    ):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                         num_classes=num_classes, embed_dim=embed_dim, depth=depth, 
                         num_heads=num_heads, drop_path_rate=drop_path_rate, weight_init = weight_init)
        self.get_feat = 'None'
        self.block_o_p = dict()
        self.neighbour_blocks = set()
        self.first_neighbour_block = 0
        self.last_neighbour_block = len(self.blocks)-1
        self.update_ln = update_ln
        self.freeze_setting = 0
        self.replaced_blocks = []
        self.GELU_array = None
        self.training = False
        self.attn_map = None

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
        elif self.get_feat == 'middle_block':
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.norm_pre(x)
            for i in range(min(self.last_neighbour_block+1,len(self.blocks))):
                x = self.blocks[i](x)
            f_middle_block = x
            # print(f'teacher.last_neighbour_block: {self.last_neighbour_block}')
            with torch.no_grad():
                if self.last_neighbour_block+1 < len(self.blocks):
                    for i in range(self.last_neighbour_block+1, len(self.blocks)):
                        x = self.blocks[i](x)
                x = self.norm(x)
                x = self.forward_head(x)
            return x, f_middle_block
        else:
            f_before = self.forward_features(x)
            x = self.forward_head(f_before)
            return x
    
    def get_blocks_to_drop(self):
        nums = len(self.blocks)
        blocks = [str(x) for x in range(nums)]
        return blocks
    
    def freeze_embed(self):
        for param in self.patch_embed.parameters():
            param.requires_grad = False
    
    def freeze_block(self, i):
        for param in self.blocks[i].parameters():
            param.requires_grad = False

    def freeze_classifier(self):
        for param in self.head.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        if not self.update_ln:
            if self.freeze_setting == 1: # freeze blocks other than neighbour blocks
                for i in range(len(self.blocks)):
                    if str(i) not in self.neighbour_blocks:
                        self.blocks[i].eval()
                    else:
                        self.blocks[i].train(mode)
            elif self.freeze_setting == 2: # freeze blocks after the last neighbour block
                for i in range(len(self.blocks)):
                    if i > self.last_neighbour_block:
                        self.blocks[i].eval()
                    else:
                        self.blocks[i].train(mode)
            elif self.freeze_setting == 5: # freeze blocks after the last neighbour block
                for i in range(len(self.blocks)):
                    if i >= self.last_neighbour_block:
                        self.blocks[i].eval()
                    else:
                        self.blocks[i].train(mode)
            elif self.freeze_setting == 3:
                for i in range(len(self.blocks)):
                    if i < self.first_neighbour_block:
                        self.blocks[i].eval()
                    else:
                        self.blocks[i].train(mode)
            elif self.freeze_setting == 4: # freeze blocks other than neighbour blocks
                self.patch_embed.eval()
                for i in range(len(self.blocks)):
                    if str(i) not in self.neighbour_blocks:
                        self.blocks[i].eval()
                    else:
                        self.blocks[i].train(mode)
        else:
            for i in range(len(self.blocks)):
                self.blocks[i].train(mode)
        return self

def _vision_transformer(
    rm_blocks,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    drop_path_rate,
    weight_init='',
    update_ln=True,
    **kwargs: Any,
) -> DeiT:
    img_size = kwargs.pop("image_size", 224)

    model = DeiT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=drop_path_rate,
        update_ln=update_ln,
        **kwargs,
    )

    if weight_init:
        load_rm_block_state_dict(model, weight_init, rm_blocks)
    
    return model

def load_rm_block_state_dict(model, raw_state_dict, rm_blocks):
    rm_count = 0
    has_count = set()
    block_o_p = dict()
    neighbour_blocks = set()
    state_dict = dict()
    raw_state_dict = {k.replace('module.',''):v for k,v in raw_state_dict.items()}
    for raw_key in raw_state_dict.keys():
        key_items = raw_key.split('.')
        if len(key_items) > 1 and key_items[0] == 'blocks' and 'total' not in key_items[-1]:
            block_key = key_items[1]
            if block_key in rm_blocks:
                if block_key not in has_count:
                    has_count.add(block_key)
                    rm_count += 1
            else:
                new_block_key = str(int(block_key) - rm_count)
                block_o_p[block_key] = new_block_key
                key_items[1] = new_block_key
                target_key = '.'.join(key_items)
                # assert target_key in state_dict, f'{raw_key} -> {target_key}'
                state_dict[target_key] = raw_state_dict[raw_key]
        elif 'total' not in key_items[-1]:
            # assert raw_key in state_dict
            state_dict[raw_key] = raw_state_dict[raw_key]
    # print(f'block_o_p: {block_o_p}')
    for rm_block in rm_blocks:
        if str(int(rm_block) - 1) in block_o_p.keys():
            neighbour_blocks.add(block_o_p[str(int(rm_block) - 1)])
        if str(int(rm_block) + 1) in block_o_p.keys():
            neighbour_blocks.add(block_o_p[str(int(rm_block) + 1)])
    # print(f'neighbour_blocks: {neighbour_blocks}')
    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    model.block_o_p = block_o_p
    if len(neighbour_blocks) != 0:
        model.neighbour_blocks = neighbour_blocks
        model.first_neighbour_block = int(min(neighbour_blocks))
        model.last_neighbour_block = int(max(neighbour_blocks))
        # print(f'first unfrozen block: {model.first_neighbour_block}, last unfrozen block: {model.last_neighbour_block}')


def rm_block_from_origin(origin_model, rm_blocks, LOG, args, pretrained=True, num_classes=1000, update_ln=True, teacher=True, **kwargs: Any) -> DeiT:
    weights = origin_model.state_dict()
    if args.model == 'deit_base_patch16_224':
        patch_size=16
        embed_dim=768
        num_heads=12
    else:
        raise ValueError("{} not found".format(args.model))
    # print(rm_blocks, len(rm_blocks))
    # assert(0)
    if args.gpu is None:
        depth = 12 - len(rm_blocks)
    else:
        depth = len(origin_model.blocks) - len(rm_blocks)
    assert(depth > 0)
    if len(rm_blocks) > 0:
        LOG.info(f'=> build deit_b_16 with {depth} layers. Remove {rm_blocks}')

    model = _vision_transformer(
        rm_blocks=rm_blocks,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=args.drop_path,
        weight_init=weights,
        update_ln=update_ln,
        **kwargs,
    )

    return model

def get_blocks_to_drop(model):
    nums = len(model.blocks)
    blocks = [str(x) for x in range(nums)]
    return blocks
