import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from .models import ViT_spock

def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "METRIC": 31,
        "TEST": 34,
        "DRAW": 38,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg

def attn_forward_wrapper(attn_obj):
    def attn_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 1:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return attn_forward

def draw_attn_heatmap(val_loader, model, fig_name, rm_blocks, attn_path, attn_map = None):
    print('drawing attention heatmap '+fig_name+'...')
    rm_blocks_list = [int(block) for block in rm_blocks.split(',')]

    for i in range(len(model.blocks)):
        if i in rm_blocks_list:
            continue
        else:
            model.blocks[i].attn.forward = attn_forward_wrapper(model.blocks[i].attn)
    
    # switch to evaluate mode
    model.eval()
    cls_weight = torch.zeros(len(model.blocks), 12, 14, 14).cuda()
    
    pbar = tqdm(range(len(val_loader)))
    for idx, (input, target) in enumerate(val_loader):
        # if idx > 10:
        #     break
        with torch.no_grad():
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            if isinstance(output, tuple):
                output = output[0]
    
            for i in range(len(model.blocks)):
                if i in rm_blocks_list:
                    continue
                else:
                    for j in range(12):
                        cls_weight[i][j] += model.blocks[i].attn.cls_attn_map[:,j,:].view(-1, 14, 14).mean(dim=0).squeeze(0).detach()
            
        msg = "drawing attention heatmap: [{}/{}]".format(idx, len(val_loader))
    
        pbar.set_description(log_msg(msg, "DRAW"))
        pbar.update()
    pbar.close()
                
    cls_weight /= len(val_loader)
    if attn_map == None:
        print("attn_map = None")
        model.attn_map = cls_weight
    cls_weight = cls_weight.cpu()
    
    array = np.zeros([12,12])
    fig, ax = plt.subplots(12, 12)
    ax = ax.flatten()
    rm = 0
    for i in range(12):
        for j in range(12):
            ax[12*i+j].axis('off')
            if i in rm_blocks_list:
                rm += 1
                heatmap = np.zeros_like(cls_weight[i][j])
            else:
                if attn_map == None:
                    heatmap = np.asarray(cls_weight[i][j])
                else:
                    attn_map = attn_map.cpu()
                    heatmap = np.asarray(np.abs(attn_map[i][j] - cls_weight[i-rm][j]))
            im = ax[12*i+j].imshow(heatmap, cmap=plt.cm.Oranges)

    fig.colorbar(im, ax=[ax[i] for i in range(144)], fraction=0.02, pad=0.05)
    plt.savefig(os.path.join(attn_path, fig_name), dpi=2000, format='eps', bbox_inches='tight')


def mlp_forward_wrapper(ffn_obj):
    def mlp_forward(x):
        x = ffn_obj.fc1(x)
        x = ffn_obj.act(x)
        ffn_obj.activation = x
        x = ffn_obj.drop1(x)
        x = ffn_obj.fc2(x)
        x = ffn_obj.drop2(x)
        return x
    return mlp_forward

def get_activation(train_loader, origin_model):
    print("get activation of GELU in original model's mlps.")
    for i in range(len(origin_model.blocks)):
        origin_model.blocks[i].mlp.forward = mlp_forward_wrapper(origin_model.blocks[i].mlp)
    
    # switch to evaluate mode
    origin_model.eval()
    activation = torch.zeros(len(origin_model.blocks), 3072).cuda()
    
    for idx, (input, target) in enumerate(train_loader):
        with torch.no_grad():
            input, target = input.cuda(), target.cuda()

            # compute output
            output = origin_model(input)
            if isinstance(output, tuple):
                output = output[0]
    
            for i in range(len(origin_model.blocks)):
                # print(origin_model.blocks[i].mlp.activation.mean(dim=0).shape)
                activation[i] += origin_model.blocks[i].mlp.activation.mean(dim=0).mean(dim=0).squeeze(0).detach()
            # assert(0)

    activation /= len(train_loader)
    activation = activation.cpu()

    # fig = plt.figure()
    GELU_array = []
    for i in range(len(origin_model.blocks)):
        # ax = fig.add_subplot(3, 4, i+1)
        actbar = np.asarray(activation[i])
        GELU_array.append(actbar.argsort()[::-1])
        # print(actbar[GELU_array])
        # assert(0)
        # positive_num.append(sum(actbar>=0))
        # positive_ind.append(np.argpartition(actbar, -positive_num[i])[-positive_num[i]:])
        # print(actbar[positive_ind[i]])
        # actbar.sort()
        # ax.bar(np.arange(3072), actbar, color='b')
        # ax.axis('off')
    origin_model.GELU_array = GELU_array
    # print(origin_model.positive_num)

    # plt.savefig(os.path.join("/opt/zhanghx/practise/img", "activation_ori.png"), dpi=400)
    

def design_save_path(load_rm_blocks, rm_blocks, args):
    if load_rm_blocks != '':
        if args.spock:
            # print("args.spock")
            pre_rm_blocks = [int(idx) for idx in load_rm_blocks.split(',')]
            new_rm_blocks = [int(idx) for idx in rm_blocks.split(',')] if rm_blocks != '' else []
            if rm_blocks != '':
                for block in new_rm_blocks:
                    pre_rm_blocks.append(int(block))
        else:
            ori2cur = dict()
            for i in range(12):
                ori2cur[i] = i
            pre_rm_blocks = [int(idx) for idx in load_rm_blocks.split(',')]
            new_rm_blocks = [int(idx) for idx in rm_blocks.split(',')] if rm_blocks != '' else []
            for pre in pre_rm_blocks:
                for key, value in ori2cur.items():
                    if key == pre:
                        ori2cur[key] = -1
                    elif key > pre:
                        ori2cur[key] -= 1
            for new in new_rm_blocks:
                for key, value in ori2cur.items():
                    if value == new:
                        pre_rm_blocks.append(key)

        pre_rm_blocks.sort()
        rm_blocks = ','.join([str(idx) for idx in pre_rm_blocks])
    
    save_path = os.path.join(args.save, 'checkpoint_rm{}_mlp{}_mlpratio{:.3f}_epoch{}_trail{}.tar'.format(
        rm_blocks, args.mlp_init, args.mlp_ratio, args.epochs, args.trail))
    
    return rm_blocks, save_path