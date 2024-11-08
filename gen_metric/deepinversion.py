# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import torch.cuda.amp as amp
import random
import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import os
import time

from utils.utils import lr_cosine_policy, lr_policy, beta_policy, mom_cosine_policy, clip, denormalize, create_folder


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module, arch_name):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.arch_name = arch_name

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        if 'vit' not in self.arch_name and 'swin' not in self.arch_name and 'deit' not in self.arch_name:
            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

            # forcing mean and variance to match between two distributions
            # other ways might work better, i.g. KL divergence
            r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
                module.running_mean.data - mean, 2)

            self.r_feature = r_feature
            # must have no output

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit, args):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    loss_var_l2 = torch.pow(torch.norm(diff1), args.beta) + torch.pow(torch.norm(diff2), args.beta) + torch.pow(torch.norm(diff3), args.beta) + torch.pow(torch.norm(diff4), args.beta)
    return loss_var_l1, loss_var_l2

def get_image_tv_loss(inputs_jit, args):
    """
    计算图像的总变分正则化。

    参数:
    x -- 输入图像, 一个 numpy 数组。
    beta -- 正则化强度参数。

    返回:
    tv -- 总变分正则化值。
    """
    x_diff_horiz = torch.diff(inputs_jit, axis=1)
    x_diff_vert = torch.diff(inputs_jit, axis=0)
    print(x_diff_horiz.shape, x_diff_vert.shape)
    tv = torch.sum(((x_diff_horiz ** 2 + x_diff_vert ** 2) ** (args.beta / 2)))
    return tv


class DeepInversionClass(object):
    def __init__(self, args, bs=84,
                 use_fp16=True,
                 net_teacher=None,
                 path="./gen_images/",
                 final_data_path="/gen_images_final/",
                 parameters=dict(),
                 setting_id=0,
                 jitter=30,
                 criterion=None,
                 coefficients=dict(),
                 network_output_function=lambda x: x,
                 hook_for_display = None):
        '''
        :param bs: batch size per GPU for image generation
        :param use_fp16: use FP16 (or APEX AMP) for model inversion, uses less memory and is faster for GPUs with Tensor Cores
        :parameter net_teacher: Pytorch model to be inverted
        :param path: path where to write temporal images and data
        :param final_data_path: path to write final images into
        :param parameters: a dictionary of control parameters:
            "resolution": input image resolution, single value, assumed to be a square, 224
            "random_label" : for classification initialize target to be random values
            "start_noise" : start from noise, def True, other options are not supported at this time
            "detach_student": if computing Adaptive DI, should we detach student?
        :param setting_id: predefined settings for optimization:
            0 - will run low resolution optimization for 1k and then full resolution for 1k;
            1 - will run optimization on high resolution for 2k
            2 - will run optimization on high resolution for 20k

        :param jitter: amount of random shift applied to image at every iteration
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L2 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
            "adi_scale" - coefficient for Adaptive DeepInversion, competition, def =0 means no competition
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        '''

        print("Deep inversion class generation")

        self.net_teacher = net_teacher

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.detach_student = parameters["detach_student"]
            self.do_flip = parameters["do_flip"]
            self.store_best_images = parameters["store_best_images"]
        else:
            self.image_resolution = 224
            self.random_label = False
            self.start_noise = True
            self.detach_student = False
            self.do_flip = True
            self.store_best_images = True

        self.setting_id = setting_id
        self.bs = bs  # batch size
        self.use_fp16 = use_fp16
        self.save_every = 100
        self.jitter = jitter
        self.criterion = criterion  # nn.CrossEntropyLoss()
        self.network_output_function = network_output_function
        do_clip = True

        if "r_feature" in coefficients:
            self.bn_reg_scale = coefficients["r_feature"]
            self.first_bn_multiplier = coefficients["first_bn_multiplier"]
            self.var_scale_l1 = coefficients["tv_l1"]
            self.var_scale_l2 = coefficients["tv_l2"]
            self.l2_scale = coefficients["l2"]
            self.lr = coefficients["lr"]
            self.main_loss_multiplier = coefficients["main_loss_multiplier"]
            self.adi_scale = coefficients["adi_scale"]
        else:
            print("Provide a dictionary with ")

        self.num_generations = 0
        self.final_data_path = final_data_path
        self.args = args

        ## Create folders for images and logs
        prefix = path
        self.prefix = prefix
        if args.new_tv == True:
            self.best_image_path = prefix + str(args.seed) + '/tv/'
            self.final_image_path = self.final_data_path + str(args.seed) + '/tv/'
        else:
            self.best_image_path = prefix + str(args.seed)
            self.final_image_path = self.final_data_path + str(args.seed)

        local_rank = torch.cuda.current_device()
        if local_rank==0:
            create_folder(prefix)
            create_folder(self.best_image_path)
            create_folder(self.final_image_path)
            # save images to folders
            # for m in range(1000):
            #     create_folder(self.final_data_path + "/s{:03d}".format(m))

        ## Create hooks for feature statistics
        self.loss_r_feature_layers = []

        for module in self.net_teacher.modules():
            if 'vit' in self.args.arch_name or 'swin' in self.args.arch_name or 'deit' in self.args.arch_name:
                if isinstance(module, nn.LayerNorm):
                    self.loss_r_feature_layers.append(DeepInversionFeatureHook(module, self.args.arch_name))
            else:
                if isinstance(module, nn.BatchNorm2d):
                    self.loss_r_feature_layers.append(DeepInversionFeatureHook(module, self.args.arch_name))

        self.hook_for_display = None
        if hook_for_display is not None:
            self.hook_for_display = hook_for_display

    def get_images(self, args, net_student=None, targets=None):
        print("get_images call")

        net_teacher = self.net_teacher
        use_fp16 = self.use_fp16
        save_every = self.save_every

        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        local_rank = torch.cuda.current_device()
        best_cost = 1e4
        criterion = self.criterion

        # setup target labels
        if targets is None:
            # only works for classification now, for other tasks need to provide target vector
            targets = torch.LongTensor([random.randint(0, 999) for _ in range(self.bs)]).to('cuda')
            if not self.random_label:
                # targets = [1, 14, 25, 36, 63, 92, 94, 107, 151, 154,
                #            189, 207, 250, 270, 277, 283, 292, 294, 309, 311,
                #            325, 340, 360, 386, 402, 403, 409, 417, 440, 468,
                #            487, 530, 574, 590, 621, 636, 670, 681, 717, 739,
                        #    752, 820, 833, 875, 946, 949, 963, 967, 980, 985]
                classes = range(0, 1000)
                targets = random.sample(classes, 50)
                targets.sort()

                targets = torch.LongTensor(targets * (int(self.bs / len(targets)))).to('cuda')

        img_original = self.image_resolution

        data_type = torch.half if use_fp16 else torch.float
        inputs = torch.randn((self.bs, 3, img_original, img_original), requires_grad=True, device='cuda',
                             dtype=data_type)
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)
        vutils.save_image(inputs, '{}/output_00001_gpu_{}.png'.format(self.best_image_path, local_rank),
                          normalize=True, scale_each=True, nrow=int(10))

        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        iteration = 0
        # assert(0)
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = 2000
            else:
                # iterations_per_layer = 1000 if not skipfirst else 2000
                iterations_per_layer = self.args.iter_num
                if self.setting_id == 2:
                    iterations_per_layer = 20000

            if lr_it==0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            if self.setting_id == 0:
                #multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True
            elif self.setting_id == 1:
                #2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True
            elif self.setting_id == 2:
                #20k normal resolution the closes to the paper experiments for ResNet50
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.9, 0.999], eps = 1e-8)
                do_clip = False

            if use_fp16:
                static_loss_scale = 256
                static_loss_scale = "dynamic"
                _, optimizer = amp.initialize([], optimizer, opt_level="O2", loss_scale=static_loss_scale)

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            start_time = time.time()
            for iteration_loc in range(iterations_per_layer):
                if iteration_loc == 100:
                    print("=> 100 iterations in {} seconds".format(time.time() - start_time))
                    # assert(0)
                iteration += 1
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # perform downsampling if needed
                if lower_res!=1:
                    inputs_jit = pooling_function(inputs)
                else:
                    inputs_jit = inputs

                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                # forward pass
                optimizer.zero_grad()
                net_teacher.zero_grad()

                outputs = net_teacher(inputs_jit)
                outputs = self.network_output_function(outputs)

                # R_cross classification loss
                loss = criterion(outputs, targets)

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit, args)
                # loss_tv = get_image_tv_loss(inputs_jit, args)

                # R_feature loss
                if 'vit' not in self.args.arch_name and 'swin' not in self.args.arch_name and 'deit' not in self.args.arch_name:
                    rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_r_feature_layers)-1)]
                    loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)])

                # R_ADI
                loss_verifier_cig = torch.zeros(1)
                if self.adi_scale!=0.0:
                    if self.detach_student:
                        outputs_student = net_student(inputs_jit).detach()
                    else:
                        outputs_student = net_student(inputs_jit)

                    T = 3.0
                    if 1:
                        T = 3.0
                        # Jensen Shanon divergence:
                        # another way to force KL between negative probabilities
                        P = nn.functional.softmax(outputs_student / T, dim=1)
                        Q = nn.functional.softmax(outputs / T, dim=1)
                        M = 0.5 * (P + Q)

                        P = torch.clamp(P, 0.01, 0.99)
                        Q = torch.clamp(Q, 0.01, 0.99)
                        M = torch.clamp(M, 0.01, 0.99)
                        eps = 0.0
                        loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                         # JS criteria - 0 means full correlation, 1 - means completely different
                        loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                    if local_rank==0:
                        if iteration % save_every==0:
                            print('loss_verifier_cig', loss_verifier_cig.item())

                # l2 loss on images
                loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                # combining losses
                if args.new_tv == True:
                    loss_aux = self.l2_scale * loss_l2 + \
                               self.var_scale_l2 * loss_tv
                else:
                    loss_aux = self.var_scale_l2 * loss_var_l2 + \
                               self.var_scale_l1 * loss_var_l1 + \
                               self.l2_scale * loss_l2
                
                
                if 'vit' not in self.args.arch_name and 'swin' not in self.args.arch_name and 'deit' not in self.args.arch_name:
                    loss_aux += self.bn_reg_scale * loss_r_feature

                if self.adi_scale!=0.0:
                    loss_aux += self.adi_scale * loss_verifier_cig

                loss = self.main_loss_multiplier * loss + loss_aux

                if local_rank==0:
                    if iteration % save_every==0:
                        print("------------iteration {}----------".format(iteration))
                        print("total loss", loss.item())
                        if 'vit' not in self.args.arch_name and 'swin' not in self.args.arch_name and 'deit' not in self.args.arch_name:
                            print("loss_r_feature", loss_r_feature.item())
                        print("main criterion", criterion(outputs, targets).item())

                        if self.hook_for_display is not None:
                            self.hook_for_display(inputs, targets)

                # do image update
                if use_fp16:
                    # optimizer.backward(loss)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

                # clip color outlayers
                if do_clip:
                    inputs.data = clip(inputs.data, self.args.arch_name, use_fp16=use_fp16)

                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()
                    # print("best cost", best_cost)

                if iteration % save_every==0 and (save_every > 0):
                    if local_rank==0:
                        vutils.save_image(inputs,
                                          '{}/output_{:05d}_gpu_{}.png'.format(self.best_image_path,
                                                                            iteration // save_every,
                                                                            local_rank),
                                          normalize=True, scale_each=True, nrow=int(10))

        if self.store_best_images:
            self.save_images(best_inputs, targets)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            #save into separate folders
            dir_to_store = '{}/s{:03d}'.format(self.final_image_path, class_id)
            if not os.path.exists(dir_to_store):
                os.makedirs(dir_to_store)
            place_to_store = '{}/s{:03d}/img_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_image_path, class_id,
                                                                                    self.num_generations, id,
                                                                                    local_rank)
            vutils.save_image(images[id], place_to_store, normalize=True, scale_each=True, nrow=int(1))

    def generate_batch(self, args, net_student=None, targets=None):
        # for ADI detach student and add put to eval mode
        net_teacher = self.net_teacher

        use_fp16 = self.use_fp16

        # fix net_student
        if not (net_student is None):
            net_student = net_student.eval()

        if targets is not None:
            targets = torch.from_numpy(np.array(targets).squeeze()).cuda()
            if use_fp16:
                targets = targets.half()

        self.get_images(args, net_student=net_student, targets=targets)

        net_teacher.eval()

        self.num_generations += 1
