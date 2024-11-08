import math
import torch
from .profile import profile
from ..models.block import BlockAttn, BlockFFN

def flops2structure(origin_model, comp_ratio, LOG, args, pointed_rm_num=-1):
    x = torch.randn(1, 3, 224, 224).cuda()
    origin_MACs, origin_Params = profile(origin_model, inputs=(x, ))

    if args.distributed:
        x = origin_model.module.patch_embed(x)
        x = origin_model.module._pos_embed(x)
        x = origin_model.module.norm_pre(x)
        block = origin_model.module.blocks[0]
    else:
        x = origin_model.patch_embed(x)
        x = origin_model._pos_embed(x)
        x = origin_model.norm_pre(x)
        block = origin_model.blocks[0]
        
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
    elif args.model in ['vit_base_patch16_224', 'deit_base_patch16_224']:
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
    attn = BlockAttn(dim=embed_dim, num_heads=num_heads).cuda()
    ffn = BlockFFN(dim=embed_dim, mlp_ratio=4).cuda()
    mlp = block.mlp

    block_MACs, block_Params = profile(block, inputs=(x, ))
    attn_MACs, attn_Params = profile(attn, inputs=(x, ))
    ffn_MACs, ffn_Params = profile(ffn, inputs=(x, ))
    mlp_MACs, mlp_Params = profile(mlp, inputs=(x, ))
    
    origin_MACs_str = f'origin={origin_MACs:.2f}G'
    block_MACs_str = f'block={block_MACs:.2f}G'
    attn_MACs_str = f'attn={attn_MACs:.2f}G'
    ffn_MACs_str = f'ffn={ffn_MACs:.2f}G'
    mlp_MACs_str = f'mlp={mlp_MACs:.2f}G'

    print(f'=> MACs: {origin_MACs_str}, {block_MACs_str}, {attn_MACs_str}, {ffn_MACs_str}, {mlp_MACs_str}')

    comp_MACs = origin_MACs * comp_ratio
    LOG.info(f'Compressed model\'s MACs: {comp_MACs:.2f}G.')
    rm_MACs = origin_MACs - comp_MACs
    if rm_MACs < attn_MACs:
        raise ValueError("Compressed MACs should be at least bigger than MACs of an attention module.")

    # use equal mlp_ratio on all replaced block(s)
    LOG.info(f'=> If use equal mlp_ratio on all replaced block(s):')
    min_rm = math.ceil(rm_MACs / (attn_MACs + mlp_MACs))
    max_rm = math.floor(rm_MACs / attn_MACs)
    
    rm_num = min_rm
    mlp_ratio = (1.0 - ((rm_MACs - rm_num * attn_MACs) / rm_num) / mlp_MACs) * 4.0
    LOG.info(f'  => you can remove {rm_num} blocks with mlp ratio {mlp_ratio}.')
    
    if pointed_rm_num != -1:
        rm_num = pointed_rm_num
        mlp_ratio = (1.0 - ((rm_MACs - rm_num * attn_MACs) / rm_num) / mlp_MACs) * 4.0
        LOG.info(f'  => you can remove {rm_num} blocks with mlp ratio {mlp_ratio}.')
    
    # allocate ratio to replaced block(s) according to their GELU>0 num
    # print(f'=> If allocate ratio to replaced block(s) according to their GELU>0 num:')
    # min_rm = math.ceil(rm_MACs / (attn_MACs + mlp_MACs))
    # max_rm = min(math.floor(rm_MACs / attn_MACs), 4)
    # for rm_num in range(min_rm, max_rm+1):
    #     GELU_num = [(768-origin_model.positive_num[i]) for i in origin_model.block_priority[:rm_num]]
    #     rm_MACs_mlp = rm_MACs - rm_num * attn_MACs
    #     rm_MACs_mlp_i = [rm_MACs_mlp * GELU_num[i] / sum(GELU_num) for i in range(rm_num)]
    #     mlp_ratio = [(1.0 - rm_MACs_mlp_i[i] / mlp_MACs) * 4.0 for i in range(rm_num)]
    #     print(f'  => you can remove block {origin_model.block_priority[:rm_num]} with mlp ratio {mlp_ratio}.')

    return rm_num, mlp_ratio


# rm 1 block: remove block [4, 3] with mlp ratio [3.0, 3.0]
# rm 2 block: remove block [4, 3, 1] with mlp ratio [1.9991319444444446, 1.9991319444444446, 1.9991319444444446]
# rm 3 block: remove block [4, 3, 1, 6] with mlp ratio [1.4986979166666665, 1.4986979166666665, 1.4986979166666665, 1.4986979166666665].