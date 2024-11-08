from typing import Type, Any, Callable, Union, List, Optional
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torchvision

# from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
# from torchvision.models import ViT_B_16_Weights
import timm
from timm.models.vision_transformer import VisionTransformer, LayerScale, DropPath, Attention
# from torchvision.models.vision_transformer import ConvStemConfig
# from torchvision.models._api import WeightsEnum
from timm.models.layers import Mlp
from .block import BlockAttn, BlockFFN

class ViT_spock(VisionTransformer):
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
            weight_init=''
    ):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                         num_classes=num_classes, embed_dim=embed_dim, depth=depth, 
                         num_heads=num_heads, drop_path_rate=drop_path_rate, weight_init = weight_init)
        self.get_feat = 'None'
        self.block_o_p = dict()
        self.neighbour_blocks = set()
        self.first_neighbour_block = -1
        self.last_neighbour_block = -1
        self.update_setting = 0
        self.replaced_blocks = []
        self.GELU_array = None
        self.training = False

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

    def freeze_classifier(self):
        for param in self.head.parameters():
            param.requires_grad = False

    # def train(self, mode=True):
    #     self.training = mode
    #     if self.update_setting != '':
    #         if self.update_setting == 1: # freeze blocks other than neighbour blocks
    #             for i in range(len(self.blocks)):
    #                 if str(i) not in self.neighbour_blocks:
    #                     self.blocks[i].eval()
    #                 else:
    #                     self.blocks[i].train(mode)
    #         elif self.update_setting == 2: # freeze blocks after the last neighbour block
    #             for i in range(len(self.blocks)):
    #                 if i > self.last_neighbour_block:
    #                     self.blocks[i].eval()
    #                 else:
    #                     self.blocks[i].train(mode)
    #         elif self.update_setting == 5: # freeze blocks after the last neighbour block
    #             for i in range(len(self.blocks)):
    #                 if i >= self.last_neighbour_block:
    #                     self.blocks[i].eval()
    #                 else:
    #                     self.blocks[i].train(mode)
    #         elif self.update_setting == 3:
    #             for i in range(len(self.blocks)):
    #                 if i < self.first_neighbour_block:
    #                     self.blocks[i].eval()
    #                 else:
    #                     self.blocks[i].train(mode)
    #         elif self.update_setting == 4: # freeze blocks other than neighbour blocks
    #             self.patch_embed.eval()
    #             for i in range(len(self.blocks)):
    #                 if str(i) not in self.neighbour_blocks:
    #                     self.blocks[i].eval()
    #                 else:
    #                     self.blocks[i].train(mode)
    #         elif self.update_setting == 'ln': # only update ln
    #             pass
    #         elif self.update_setting == 'ffn_bias': # only update ln
    #             pass
    #     else:
    #         for i in range(len(self.blocks)):
    #             self.blocks[i].train(mode)
    #     return self

def _vision_transformer(
        args,
        load_rm_blocks, 
        rm_blocks,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        drop_path_rate,
        GELU_array,
        weight_init='',
        **kwargs: Any,
    ) -> ViT_spock:
    img_size = kwargs.pop("image_size", 224)

    model = ViT_spock(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )

    if weight_init:
        load_rm_block_state_dict(args, model, weight_init, load_rm_blocks, rm_blocks, embed_dim, GELU_array)
    
    return model

def load_rm_block_state_dict(args, model, raw_state_dict, load_rm_blocks, rm_blocks, embed_dim, GELU_array):
    raw_state_dict = {k.replace('module.',''):v for k,v in raw_state_dict.items()}

    def build_mlp(block, load=False):
        # mlp_ratio = mlp_dim[int(block)] / embed_dim
        # select_array = mlp_ind[int(block)]
        # select_array.sort()
        # print(f"Dim of mlp of new block {block}: {mlp_dim[int(block)]}")
        mlp_init = args.mlp_init
        if load:
            mlp_init = 'rand_sample'
        mlp_ratio = args.mlp_ratio
        if mlp_init == 'rand_sample':
            rand_array = np.arange(raw_state_dict['blocks.'+block+'.mlp.fc1.weight'].shape[0])
            # print(rand_array)
            np.random.shuffle(rand_array)
            select_array = rand_array[:int(embed_dim * mlp_ratio)]
            select_array.sort()
        elif mlp_init == 'GELU_sample':
            # print(len(GELU_array))
            select_array = GELU_array[int(block)][:int(embed_dim * mlp_ratio)].copy()
            select_array.sort()
            # print(select_array)
        else:
            raise ValueError(f'{args.mlp_init} not found.')
        
        simple_block = BlockFFN(dim=embed_dim, mlp_ratio=mlp_ratio)
        
        state_dict = dict()
        state_dict['norm2.weight'] = raw_state_dict['blocks.'+block+'.norm2.weight']
        state_dict['norm2.bias'] = raw_state_dict['blocks.'+block+'.norm2.bias']
        state_dict['mlp.fc1.weight'] = raw_state_dict['blocks.'+block+'.mlp.fc1.weight'][select_array,:]
        state_dict['mlp.fc1.bias'] = raw_state_dict['blocks.'+block+'.mlp.fc1.bias'][select_array]
        state_dict['mlp.fc2.weight'] = raw_state_dict['blocks.'+block+'.mlp.fc2.weight'][:,select_array]
        state_dict['mlp.fc2.bias'] = raw_state_dict['blocks.'+block+'.mlp.fc2.bias']
        
        simple_block.load_state_dict(state_dict)
        return simple_block
    
    def rm_attn(block):
        simple_block = BlockFFN(dim=embed_dim)
        
        state_dict = dict()
        state_dict['norm2.weight'] = raw_state_dict['blocks.'+block+'.norm2.weight']
        state_dict['norm2.bias'] = raw_state_dict['blocks.'+block+'.norm2.bias']
        state_dict['mlp.fc1.weight'] = raw_state_dict['blocks.'+block+'.mlp.fc1.weight']
        state_dict['mlp.fc1.bias'] = raw_state_dict['blocks.'+block+'.mlp.fc1.bias']
        state_dict['mlp.fc2.weight'] = raw_state_dict['blocks.'+block+'.mlp.fc2.weight']
        state_dict['mlp.fc2.bias'] = raw_state_dict['blocks.'+block+'.mlp.fc2.bias']
        
        simple_block.load_state_dict(state_dict)
        return simple_block
    
    def rm_ffn(block):
        simple_block = BlockAttn(dim=embed_dim, num_heads=12, qkv_bias=True)
        
        state_dict = dict()
        state_dict['norm1.weight'] = raw_state_dict['blocks.'+block+'.norm1.weight']
        state_dict['norm1.bias'] = raw_state_dict['blocks.'+block+'.norm1.bias']
        state_dict['attn.qkv.weight'] = raw_state_dict['blocks.'+block+'.attn.qkv.weight']
        state_dict['attn.qkv.bias'] = raw_state_dict['blocks.'+block+'.attn.qkv.bias']
        state_dict['attn.proj.weight'] = raw_state_dict['blocks.'+block+'.attn.proj.weight']
        state_dict['attn.proj.bias'] = raw_state_dict['blocks.'+block+'.attn.proj.bias']

        # print("in rm_ffn:", state_dict['attn.qkv.weight'])
        
        simple_block.load_state_dict(state_dict)
        return simple_block
    
    def build_random_mlp(block, load=False):
        mlp_ratio = args.mlp_ratio
        
        simple_block = BlockFFN(dim=embed_dim, mlp_ratio=mlp_ratio)
        
        simple_block.norm2.weight.data.copy_(raw_state_dict['blocks.'+block+'.norm2.weight'])
        simple_block.norm2.bias.data.copy_(raw_state_dict['blocks.'+block+'.norm2.bias'])
        
        return simple_block
    
    def build_identity(block):
        simple_block = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        simple_block.weight.data.copy_(torch.eye(embed_dim))
        return simple_block

    if args.spock in ['spock', 'once', 'progressive', 'replace']:
        if args.mlp_init == 'rand_init':
            build_simpleblock = build_random_mlp
        else:
            build_simpleblock = build_mlp
    elif args.spock == 'rm_attn':
        build_simpleblock = rm_attn
    elif args.spock == 'rm_ffn':
        build_simpleblock = rm_ffn
    elif args.spock == 'identity':
        build_simpleblock = build_identity
    else:
        raise ValueError("{} not found".format(args.spock))

    for block in load_rm_blocks:
        model.blocks[int(block)] = build_simpleblock(block, load=True)
        model.replaced_blocks.append(int(block))

    for block in rm_blocks:
        # assert(block not in model.replaced_blocks)
        model.blocks[int(block)] = build_simpleblock(block)
        # model.blocks[int(block)] = build_identity(block)
        model.replaced_blocks.append(int(block))
    
    model.first_neighbour_block = max(min(model.replaced_blocks)-1, 0)
    model.last_neighbour_block = min(max(model.replaced_blocks)+1, 11)
    if args.update_setting == 'inter':
        rm = [int(block) for block in rm_blocks]
        model.first_neighbour_block = max(min(rm)-2, 0)
        model.last_neighbour_block = min(max(rm)+2, 11)
        # print(model.first_neighbour_block, model.last_neighbour_block)
    
    new_state_dict = dict()
    # print(raw_state_dict.keys())
    for raw_key in raw_state_dict.keys():
        key_items = raw_key.split('.')
        if len(key_items) > 1 and key_items[0] == 'blocks':
            if key_items[1] not in rm_blocks and 'total' not in key_items[-1]:
                # print(f"use raw key in model {raw_key}")
                new_state_dict[raw_key] = raw_state_dict[raw_key]
        elif 'total' not in key_items[-1]:
            new_state_dict[raw_key] = raw_state_dict[raw_key]
    
    for new_key in model.state_dict().keys():
        key_items = new_key.split('.')
        if len(key_items) > 1 and key_items[0] == 'blocks':
            if key_items[1] in rm_blocks and 'total' not in key_items[-1]:
                # print(f"use new key in model {new_key}")
                new_state_dict[new_key] = model.state_dict()[new_key]

    model.load_state_dict(new_state_dict)


def replace_block_from_origin(origin_model, load_rm_blocks, rm_blocks, LOG, args, pretrained=True, num_classes=1000, teacher=True, **kwargs: Any) -> ViT_spock:
    weights = origin_model.state_dict()
    if args.model == 'vit_tiny_patch16_224':
        patch_size=16
        embed_dim=192
        num_heads=3
    elif args.model == 'vit_small_patch16_224':
        patch_size=16
        embed_dim=384
        num_heads=6
    elif args.model == 'vit_small_patch32_224':
        patch_size=32
        embed_dim=384
        num_heads=6
    elif args.model == 'vit_base_patch16_224':
        patch_size=16
        embed_dim=768
        num_heads=12
    elif args.model == 'vit_base_patch32_224':
        patch_size=32
        embed_dim=768
        num_heads=12
    elif args.model == 'vit_large_patch16_224':
        patch_size=16
        embed_dim=1024
        num_heads=16
    else:
        raise ValueError("{} not found".format(args.model))
    
    if args.gpu == None:
        depth = 12
    else:
        depth = len(origin_model.blocks)
    assert(depth > 0)
    if len(rm_blocks) > 0:
        LOG.info(f'=> build vit_b_16 with {depth} layers. Remove {rm_blocks}')

    model = _vision_transformer(
        args=args,
        load_rm_blocks=load_rm_blocks,
        rm_blocks=rm_blocks,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=args.drop_path,
        GELU_array = origin_model.GELU_array,
        weight_init=weights,
        **kwargs,
    )

    return model

def get_blocks_to_replace(model):
    nums = len(model.blocks)
    blocks = [str(x) for x in range(nums) if x not in model.replaced_blocks]
    return blocks
