import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from timm.models.layers import Mlp

class AdaptorWarp(nn.Module):
    def __init__(self, model) -> None:
        super(AdaptorWarp, self).__init__()
        self.model = model

        self.named_modules_dict = dict(model.named_modules())
        self.module2preadaptors = dict()
        self.name2preadaptors = dict()
        self.prehandles = dict()
        self.module2afteradaptors = dict()
        self.name2afteradaptors = dict()
        self.afterhandles = dict()


    def add_preconv_for_conv(self, name, convmodule=None, preconv=None):
        assert name in self.named_modules_dict
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name in self.name2preadaptors:
            print("Not insert because {} have defined pre-conv".format(name))
            return None
        if preconv is None:
            v = convmodule.in_channels
            preconv = nn.Conv2d(v, v, kernel_size=1, stride=1, padding=0, bias=False).cuda()
            preconv.weight.data.copy_(torch.eye(v).view(v, v, 1, 1))
        self.module2preadaptors[convmodule] = preconv
        self.name2preadaptors[name] = preconv
        def hook(module, input):
            preconv = self.module2preadaptors[module]
            return preconv(input[0])
        self.prehandles[convmodule] = convmodule.register_forward_pre_hook(hook)
        print("=> add pre-conv on {}".format(name))
        return preconv

    def reset_preconv_for_conv(self, name, convmodule=None):
        assert name in self.named_modules_dict
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name not in self.name2preadaptors:
            print("WARN: Can not reset, {} have not defined pre-conv".format(name))
            return
        preconv = self.name2preadaptors[name]
        v = convmodule.in_channels
        preconv.weight.data.copy_(torch.eye(v).view(v, v, 1, 1))
        return preconv


    def remove_preconv_for_conv(self, name, convmodule=None, absorb=False):
        assert name in self.named_modules_dict
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name not in self.name2preadaptors.keys():
            if absorb:
                print("WARN: cannot absorb because not find pre-conv on {}".format(name))
            return
        if absorb:
            print("=> absorb pre-conv on {}".format(name))
            pw = self.name2preadaptors[name].weight.data
            weight = convmodule.weight.data
            w = weight.permute(2, 3, 0, 1)
            # w: 3 x 3 x out x in 
            # use double type
            new_weight = torch.matmul(w.double(), pw.squeeze().double())
            new_weight = new_weight.float().permute(2, 3, 0, 1)
            convmodule.weight.data.copy_(new_weight)
        # remove the hook
        self.prehandles[convmodule].remove()
        self.name2preadaptors.pop(name)
        print("=> remove pre-conv on {}".format(name))


    def add_all_preconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.add_preconv_for_conv(name, module)

    def reset_all_preconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.reset_preconv_for_conv(name, module)

    def absorb_all_preconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.remove_preconv_for_conv(name, module, absorb=True)

    def remove_all_preconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.remove_preconv_for_conv(name, module)
                

    def add_afterconv_for_conv(self, name, convmodule=None, afterconv=None):
        assert name in self.named_modules_dict, "{} not in {}".format(
            name, self.named_modules_dict.keys())
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name in self.name2afteradaptors:
            print("Not insert because {} have defined after-conv".format(name))
            return None
        if afterconv is None:
            v = convmodule.out_channels
            afterconv = nn.Conv2d(v, v, kernel_size=1, stride=1, padding=0, bias=False).cuda()
            afterconv.weight.data.copy_(torch.eye(v).view(v, v, 1, 1))
        self.module2afteradaptors[convmodule] = afterconv
        self.name2afteradaptors[name] = afterconv
        def hook(module, input, output):
            afterconv = self.module2afteradaptors[module]
            return afterconv(output)
        self.afterhandles[convmodule] = convmodule.register_forward_hook(hook)
        print("=> add after-conv on {}".format(name))
        return afterconv

    def reset_afterconv_for_conv(self, name, convmodule=None):
        assert name in self.named_modules_dict
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name not in self.name2afteradaptors:
            print("WARN: Can not reset, {} have not defined after-conv".format(name))
            return
        afterconv = self.name2afteradaptors[name]
        v = convmodule.out_channels
        afterconv.weight.data.copy_(torch.eye(v).view(v, v, 1, 1))
        return afterconv

    def remove_afterconv_for_conv(self, name, convmodule=None, absorb=False):
        assert name in self.named_modules_dict, "{} not in {}".format(name, self.named_modules_dict.keys())
        module = self.named_modules_dict[name]
        if convmodule is None:
            convmodule = module
        else:
            assert module == convmodule
        assert isinstance(convmodule, torch.nn.Conv2d)
        if name not in self.name2afteradaptors.keys():
            if absorb:
                print("WARN: cannot absorb because not find after-conv on {}".format(name))
            return
        if absorb:
            print("=> absorb after-conv on {}".format(name))
            pw = self.name2afteradaptors[name].weight.data
            weight = convmodule.weight.data
            w = weight.permute(2, 3, 0, 1)
            # w: 3 x 3 x out x in 
            # use double type
            new_weight = torch.matmul(pw.squeeze().double(), w.double())
            new_weight = new_weight.float().permute(2, 3, 0, 1)
            convmodule.weight.data.copy_(new_weight)
            if convmodule.bias is not None:
                new_bias = torch.matmul(pw.double().squeeze(), convmodule.bias.data.double().unsqueeze(1))
                convmodule.bias.data.copy_(new_bias.float().flatten())
        # remove the hook
        self.afterhandles[convmodule].remove()
        self.name2afteradaptors.pop(name)
        print("=> remove after-conv on {}".format(name))

    def add_all_afterconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.add_afterconv_for_conv(name, module)

    def reset_all_afterconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.reset_afterconv_for_conv(name, module)

    def absorb_all_afterconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.remove_afterconv_for_conv(name, module, absorb=True)

    def remove_all_afterconv(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.remove_afterconv_for_conv(name, module)



    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        return self.model(x)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def freeze_classifier(self):
        self.model.freeze_classifier()


    def cuda(self):
        for key in self.name2preadaptors.keys():
            self.name2preadaptors[key].cuda()
        for key in self.name2afteradaptors.keys():
            self.name2afteradaptors[key].cuda()
        return super(AdaptorWarp, self).cuda()
    

    def add_preadaptor_for_linear(self, name, adaptor_type='linear', adaptormodule=None, preadaptor=None, LOG=None):
        assert name in self.named_modules_dict
        module = self.named_modules_dict[name]
        if adaptormodule is None:
            adaptormodule = module
        else:
            assert module == adaptormodule
        assert isinstance(adaptormodule, torch.nn.Linear)
        if name in self.name2preadaptors:
            LOG.info("Not insert because {} have defined pre-conv".format(name))
            return None
        if preadaptor is None:
            v = adaptormodule.in_features
            if adaptor_type == 'linear':
                preadaptor = nn.Linear(in_features=v, out_features=v).cuda()
                preadaptor.weight.data.copy_(torch.eye(v))
            elif adaptor_type == 'mlp':
                mlp_ratio = 1.
                preadaptor = Mlp(in_features=v, hidden_features=int(v * mlp_ratio), out_features=v).cuda()
                preadaptor.fc1.weight.data.copy_(torch.eye(v))
                preadaptor.fc2.weight.data.copy_(torch.eye(v))
        self.module2preadaptors[adaptormodule] = preadaptor
        self.name2preadaptors[name] = preadaptor
        def hook(module, input):
            preadaptor = self.module2preadaptors[module]
            return preadaptor(input[0])
        self.prehandles[adaptormodule] = adaptormodule.register_forward_pre_hook(hook)
        LOG.info("=> add pre-adaptor on {}".format(name))
        return preadaptor
    
    def add_afteradaptor_for_linear(self, name, adaptor_type='linear', adaptormodule=None, afteradaptor=None, LOG=None):
        assert name in self.named_modules_dict, "{} not in {}".format(
            name, self.named_modules_dict.keys())
        module = self.named_modules_dict[name]
        if adaptormodule is None:
            adaptormodule = module
        else:
            assert module == adaptormodule
        assert isinstance(adaptormodule, torch.nn.Linear)
        if name in self.name2afteradaptors:
            LOG.info("Not insert because {} have defined after-conv".format(name))
            return None
        if afteradaptor is None:
            v = adaptormodule.out_features
            if adaptor_type == 'linear':
                afteradaptor = nn.Linear(in_features=v, out_features=v).cuda()
                afteradaptor.weight.data.copy_(torch.eye(v))
            elif adaptor_type == 'mlp':
                mlp_ratio = 1.
                afteradaptor = Mlp(in_features=v, hidden_features=int(v * mlp_ratio), out_features=v).cuda()
                afteradaptor.fc1.weight.data.copy_(torch.eye(v))
                afteradaptor.fc2.weight.data.copy_(torch.eye(v))
        self.module2afteradaptors[adaptormodule] = afteradaptor
        self.name2afteradaptors[name] = afteradaptor
        def hook(module, input, output):
            afteradaptor = self.module2afteradaptors[module]
            return afteradaptor(output)
        self.afterhandles[adaptormodule] = adaptormodule.register_forward_hook(hook)
        LOG.info("=> add after-adaptor on {}".format(name))
        return afteradaptor