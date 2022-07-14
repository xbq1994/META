# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
import pdb
try:
    from apex import parallel
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run model with syncBN")

__all__ = ["IBN", "get_norm"]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)

class BatchNorm1(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)

class BatchNorm2(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)

class BatchNorm3(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)
        
class BatchNorm4(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)

class SyncBatchNorm(parallel.SyncBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class IBN(nn.Module):
    def __init__(self, planes, bn_norm, **kwargs):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = get_norm(bn_norm, half2, **kwargs)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out



class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits=1, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            self.running_mean = self.running_mean.repeat(self.num_splits)
            self.running_var = self.running_var.repeat(self.num_splits)
            outputs = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0)
            return outputs
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


class FrozenBatchNorm(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    Other pre-trained backbone models may contain all 4 parameters.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

class SwitchNorm2d_1(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight_1 = nn.Parameter(torch.ones(num_features),requires_grad=True)
        self.bias_1 = nn.Parameter(torch.zeros(num_features),requires_grad=True)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.register_buffer('mean_in', torch.zeros(64,num_features))
        self.register_buffer('var_in', torch.zeros(64,num_features))
        self.use_train = True
        self.reset_parameters()

    def set_eval(self):
        self.use_train = False
        self.weight_1.requires_grad=False
        self.bias_1.requires_grad=False
        
    def set_train(self):
        self.use_train = True
        self.weight_1.requires_grad_(True)
        self.bias_1.requires_grad_(True)
                
    def reset_parameters(self):

        self.running_mean.zero_()
        self.running_var.zero_()
        self.mean_in.zero_()
        self.var_in.zero_()
        
        if self.last_gamma:
            self.weight_1.data.fill_(0)
        else:
            self.weight_1.data.fill_(1)
        self.bias_1.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        mean_in = x.mean(-1)
        var_in = x.var(-1)
        temp = var_in + mean_in ** 2

        if self.training and self.use_train:
            mean_bn = mean_in.mean(0)
            var_bn = temp.mean(0) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean, requires_grad=False)
            var_bn = torch.autograd.Variable(self.running_var, requires_grad=False)

        mean = mean_bn.reshape(1,-1,1)
        var = var_bn.reshape(1,-1,1)
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)

        self.mean_in = mean_in
        self.var_in = var_in
        
        if self.training and self.use_train:
            return x * self.weight_1.reshape(1,-1,1,1) + self.bias_1.reshape(1,-1,1,1)
        else:
            weight = torch.autograd.Variable(self.weight_1, requires_grad=False)
            bias = torch.autograd.Variable(self.bias_1, requires_grad=False)
            return x * weight.reshape(1,-1,1,1) + bias.reshape(1,-1,1,1)

class SwitchNorm2d_2(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight_2 = nn.Parameter(torch.ones(num_features),requires_grad=True)
        self.bias_2 = nn.Parameter(torch.zeros(num_features),requires_grad=True)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.register_buffer('mean_in', torch.zeros(64,num_features))
        self.register_buffer('var_in', torch.zeros(64,num_features))
        self.use_train = True
        self.reset_parameters()

    def set_eval(self):
        self.use_train = False
        self.weight_2.requires_grad=False
        self.bias_2.requires_grad=False
        
    def set_train(self):
        self.use_train = True
        self.weight_2.requires_grad_(True)
        self.bias_2.requires_grad_(True)
                
    def reset_parameters(self):

        self.running_mean.zero_()
        self.running_var.zero_()
        self.mean_in.zero_()
        self.var_in.zero_()
        
        if self.last_gamma:
            self.weight_2.data.fill_(0)
        else:
            self.weight_2.data.fill_(1)
        self.bias_2.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        mean_in = x.mean(-1)
        var_in = x.var(-1)
        temp = var_in + mean_in ** 2

        if self.training and self.use_train:
            mean_bn = mean_in.mean(0)
            var_bn = temp.mean(0) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean, requires_grad=False)
            var_bn = torch.autograd.Variable(self.running_var, requires_grad=False)

        mean = mean_bn.reshape(1,-1,1)
        var = var_bn.reshape(1,-1,1)
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)

        self.mean_in = mean_in
        self.var_in = var_in
        
        if self.training and self.use_train:
            return x * self.weight_2.reshape(1,-1,1,1) + self.bias_2.reshape(1,-1,1,1)
        else:
            weight = torch.autograd.Variable(self.weight_2, requires_grad=False)
            bias = torch.autograd.Variable(self.bias_2, requires_grad=False)
            return x * weight.reshape(1,-1,1,1) + bias.reshape(1,-1,1,1)

class SwitchNorm2d_3(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight_3 = nn.Parameter(torch.ones(num_features),requires_grad=True)
        self.bias_3 = nn.Parameter(torch.zeros(num_features),requires_grad=True)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.register_buffer('mean_in', torch.zeros(64,num_features))
        self.register_buffer('var_in', torch.zeros(64,num_features))
        self.use_train = True
        self.reset_parameters()

    def set_eval(self):
        self.use_train = False
        self.weight_3.requires_grad=False
        self.bias_3.requires_grad=False
        
    def set_train(self):
        self.use_train = True
        self.weight_3.requires_grad_(True)
        self.bias_3.requires_grad_(True)
                
    def reset_parameters(self):

        self.running_mean.zero_()
        self.running_var.zero_()
        self.mean_in.zero_()
        self.var_in.zero_()
        
        if self.last_gamma:
            self.weight_3.data.fill_(0)
        else:
            self.weight_3.data.fill_(1)
        self.bias_3.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        mean_in = x.mean(-1)
        var_in = x.var(-1)
        temp = var_in + mean_in ** 2

        if self.training and self.use_train:
            mean_bn = mean_in.mean(0)
            var_bn = temp.mean(0) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean, requires_grad=False)
            var_bn = torch.autograd.Variable(self.running_var, requires_grad=False)

        mean = mean_bn.reshape(1,-1,1)
        var = var_bn.reshape(1,-1,1)
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)

        self.mean_in = mean_in
        self.var_in = var_in
        
        if self.training and self.use_train:
            return x * self.weight_3.reshape(1,-1,1,1) + self.bias_3.reshape(1,-1,1,1)
        else:
            weight = torch.autograd.Variable(self.weight_3, requires_grad=False)
            bias = torch.autograd.Variable(self.bias_3, requires_grad=False)
            return x * weight.reshape(1,-1,1,1) + bias.reshape(1,-1,1,1)

class SwitchNorm2d_4(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight_4 = nn.Parameter(torch.ones(num_features),requires_grad=True)
        self.bias_4 = nn.Parameter(torch.zeros(num_features),requires_grad=True)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.register_buffer('mean_in', torch.zeros(64,num_features))
        self.register_buffer('var_in', torch.zeros(64,num_features))
        self.use_train = True
        self.reset_parameters()

    def set_eval(self):
        self.use_train = False
        self.weight_4.requires_grad=False
        self.bias_4.requires_grad=False
        
    def set_train(self):
        self.use_train = True
        self.weight_4.requires_grad_(True)
        self.bias_4.requires_grad_(True)
                
    def reset_parameters(self):

        self.running_mean.zero_()
        self.running_var.zero_()
        self.mean_in.zero_()
        self.var_in.zero_()
        
        if self.last_gamma:
            self.weight_4.data.fill_(0)
        else:
            self.weight_4.data.fill_(1)
        self.bias_4.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        mean_in = x.mean(-1)
        var_in = x.var(-1)
        temp = var_in + mean_in ** 2

        if self.training and self.use_train:
            mean_bn = mean_in.mean(0)
            var_bn = temp.mean(0) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean, requires_grad=False)
            var_bn = torch.autograd.Variable(self.running_var, requires_grad=False)

        mean = mean_bn.reshape(1,-1,1)
        var = var_bn.reshape(1,-1,1)
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)

        self.mean_in = mean_in
        self.var_in = var_in
        
        if self.training and self.use_train:
            return x * self.weight_4.reshape(1,-1,1,1) + self.bias_4.reshape(1,-1,1,1)
        else:
            weight = torch.autograd.Variable(self.weight_4, requires_grad=False)
            bias = torch.autograd.Variable(self.bias_4, requires_grad=False)
            return x * weight.reshape(1,-1,1,1) + bias.reshape(1,-1,1,1)
            
                        
def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, GhostBN, FrozenBN, GN or SyncBN;
            or a callable that thakes a channel number and returns
            the normalization layer as a nn.Module
        out_channels: number of channels for normalization layer

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm,
            "syncBN": SyncBatchNorm,
            "GhostBN": GhostBatchNorm,
            "FrozenBN": FrozenBatchNorm,
            "GN": lambda channels, **args: nn.GroupNorm(32, channels),
            "switchnorm": SwitchNorm2d_1,
        }[norm]
    return norm(out_channels, **kwargs)
