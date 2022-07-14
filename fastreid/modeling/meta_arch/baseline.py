# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from torch.nn import functional as F
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
import pdb




@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        backbone = build_backbone(cfg)
        self.backbone = backbone[0]
        self.backbone_agg = backbone[1]
        
        self.heads_agg = build_heads(cfg)
        self.heads_F_final = build_heads(cfg)
        self.heads_F_finals = build_heads(cfg)
        self.heads_F_final_Expert1 = build_heads(cfg, num_classes=cfg.DATASETS.CLASSES[0])
        self.heads_F_final_Expert2 = build_heads(cfg, num_classes=cfg.DATASETS.CLASSES[1])
        self.heads_F_final_Expert3 = build_heads(cfg, num_classes=cfg.DATASETS.CLASSES[2])
        self.heads_Expert1 = build_heads(cfg, num_classes=cfg.DATASETS.CLASSES[0])
        self.heads_Expert2 = build_heads(cfg, num_classes=cfg.DATASETS.CLASSES[1])
        self.heads_Expert3 = build_heads(cfg, num_classes=cfg.DATASETS.CLASSES[2])
        
        self.affine1_Expert1 = nn.Linear(53, 512)
        self.affine2_Expert1 = nn.Linear(512, 1)
        nn.init.kaiming_normal_(self.affine1_Expert1.weight, mode='fan_out')
        nn.init.constant_(self.affine1_Expert1.bias, 0)
        nn.init.kaiming_normal_(self.affine2_Expert1.weight, mode='fan_out')
        nn.init.constant_(self.affine2_Expert1.bias, 0)

        self.affine1_Expert2 = nn.Linear(53, 512)
        self.affine2_Expert2 = nn.Linear(512, 1)
        nn.init.kaiming_normal_(self.affine1_Expert2.weight, mode='fan_out')
        nn.init.constant_(self.affine1_Expert2.bias, 0)
        nn.init.kaiming_normal_(self.affine2_Expert2.weight, mode='fan_out')
        nn.init.constant_(self.affine2_Expert2.bias, 0)

        self.affine1_Expert3 = nn.Linear(53, 512)
        self.affine2_Expert3 = nn.Linear(512, 1)
        nn.init.kaiming_normal_(self.affine1_Expert3.weight, mode='fan_out')
        nn.init.constant_(self.affine1_Expert3.bias, 0)
        nn.init.kaiming_normal_(self.affine2_Expert3.weight, mode='fan_out')
        nn.init.constant_(self.affine2_Expert3.bias, 0)

        self.bn_mean_Expert1 = []
        self.bn_mean_Expert2 = []   
        self.bn_mean_Expert3 = []
        self.bn_var_Expert1 = []
        self.bn_var_Expert2 = []   
        self.bn_var_Expert3 = []
        self.in_mean_Expert1 = []
        self.in_mean_Expert2 = []   
        self.in_mean_Expert3 = []        
        self.in_var_Expert1 = []
        self.in_var_Expert2 = []   
        self.in_var_Expert3 = [] 

        self.mse_loss = nn.MSELoss(reduce=True, size_average=True)
        self.ranking_loss = nn.MarginRankingLoss(margin=0.0)
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs,iters=0):
        EMD_Expert1=0
        EMD_Expert2=0
        EMD_Expert3=0
        
        x, data_name = self.preprocess_image(batched_inputs)

        x_Expert1 = self.backbone(x,ids=1, data_name=data_name)
        self.get_in_Expert1_list() 
        
        x_Expert2 = self.backbone(x,ids=2, data_name=data_name)
        self.get_in_Expert2_list() 
        
        x_Expert3 = self.backbone(x,ids=3, data_name=data_name) 
        self.get_in_Expert3_list() 
        
        x_agg = self.backbone(x,ids=4, data_name=data_name)
        x_agg = self.backbone_agg(x_agg)

        self.get_bn_list()
        
        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            
            targets_agg = batched_inputs["targets"].to(self.device)
            targets_expert = batched_inputs["targets_expert"].to(self.device)
            #print('self.device:',self.device,'data_name:',data_name,'img_path:',batched_inputs['img_paths'])
            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            
            if data_name == 'Expert1': 
                x_Expert1_output = self.heads_Expert1(x_Expert1, targets_expert)
                x_Expert2_output = self.heads_Expert2(x_Expert2)
                x_Expert3_output = self.heads_Expert3(x_Expert3)
            elif data_name == 'Expert2': 
                x_Expert1_output = self.heads_Expert1(x_Expert1)
                x_Expert2_output = self.heads_Expert2(x_Expert2, targets_expert)
                x_Expert3_output = self.heads_Expert3(x_Expert3)
            elif data_name == 'Expert3': 
                x_Expert1_output = self.heads_Expert1(x_Expert1)
                x_Expert2_output = self.heads_Expert2(x_Expert2)
                x_Expert3_output = self.heads_Expert3(x_Expert3, targets_expert)

            x_agg_output = self.heads_agg(x_agg, targets_agg)             
                            

            if data_name == 'Expert1':         
                
                for i in range(len(self.bn_mean_Expert2)):
                    EMD_ = self.EMD(self.bn_mean_Expert2[i],self.bn_var_Expert2[i],self.in_mean_Expert2[i],self.in_var_Expert2[i])
                    
                    if i==0:
                        EMD_Expert2 = EMD_
                    else:
                        EMD_Expert2 = torch.cat((EMD_Expert2,EMD_),dim=-1)
                   
                
                for i in range(len(self.bn_mean_Expert3)):
                    EMD_ = self.EMD(self.bn_mean_Expert3[i],self.bn_var_Expert3[i],self.in_mean_Expert3[i],self.in_var_Expert3[i])
                    if i==0:
                        EMD_Expert3 = EMD_
                    else:
                        EMD_Expert3 = torch.cat((EMD_Expert3,EMD_),dim=-1)  

                n1 = self.affine1_Expert2(EMD_Expert2)
                n1 = self.affine2_Expert2(n1)
                n2 = self.affine1_Expert3(EMD_Expert3)
                n2 = self.affine2_Expert3(n2)                
                aff = F.softmax(torch.cat((n1,n2),dim=-1),dim=-1)
                F_final = aff[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_Expert2.detach()+aff[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_Expert3.detach()
                
                F_final_output = self.heads_F_final(F_final, targets_agg)
                losses = self.losses(x_Expert1_output, F_final_output, x_agg_output, targets_agg, targets_expert,iters)
                return losses
            
            elif data_name == 'Expert2':
                
                for i in range(len(self.bn_mean_Expert1)):
                    EMD_ = self.EMD(self.bn_mean_Expert1[i],self.bn_var_Expert1[i],self.in_mean_Expert1[i],self.in_var_Expert1[i])
                    if i==0:
                        EMD_Expert1 = EMD_
                    else:
                        EMD_Expert1 = torch.cat((EMD_Expert1,EMD_),dim=-1)   
                for i in range(len(self.bn_mean_Expert3)):
                    EMD_ = self.EMD(self.bn_mean_Expert3[i],self.bn_var_Expert3[i],self.in_mean_Expert3[i],self.in_var_Expert3[i])
                    if i==0:
                        EMD_Expert3 = EMD_
                    else:
                        EMD_Expert3 = torch.cat((EMD_Expert3,EMD_),dim=-1)                       
                n1 = self.affine1_Expert1(EMD_Expert1)
                n1 = self.affine2_Expert1(n1)
                n2 = self.affine1_Expert3(EMD_Expert3)
                n2 = self.affine2_Expert3(n2)  
                aff = F.softmax(torch.cat((n1,n2),dim=-1),dim=-1)
                F_final = aff[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_Expert1.detach()+aff[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_Expert3.detach()

                F_final_output = self.heads_F_final(F_final, targets_agg)           
                losses = self.losses(x_Expert2_output, F_final_output, x_agg_output, targets_agg, targets_expert,iters)
                return losses
  
            elif data_name == 'Expert3':
                
                for i in range(len(self.bn_mean_Expert1)):
                    EMD_ = self.EMD(self.bn_mean_Expert1[i],self.bn_var_Expert1[i],self.in_mean_Expert1[i],self.in_var_Expert1[i])
                    if i==0:
                        EMD_Expert1 = EMD_
                    else:
                        EMD_Expert1 = torch.cat((EMD_Expert1,EMD_),dim=-1)   
                for i in range(len(self.bn_mean_Expert2)):
                    EMD_ = self.EMD(self.bn_mean_Expert2[i],self.bn_var_Expert2[i],self.in_mean_Expert2[i],self.in_var_Expert2[i])
                    if i==0:
                        EMD_Expert2 = EMD_
                    else:
                        EMD_Expert2 = torch.cat((EMD_Expert2,EMD_),dim=-1)            

                n1 = self.affine1_Expert1(EMD_Expert1)
                n1 = self.affine2_Expert1(n1)
                n2 = self.affine1_Expert2(EMD_Expert2)
                n2 = self.affine2_Expert2(n2) 
                aff = F.softmax(torch.cat((n1,n2),dim=-1),dim=-1)
                F_final = aff[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_Expert1.detach()+aff[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_Expert2.detach()

                F_final_output = self.heads_F_final(F_final, targets_agg)
                losses = self.losses(x_Expert3_output, F_final_output, x_agg_output, targets_agg, targets_expert,iters)
                return losses
               
        else:
            x_Expert1_output = self.heads_Expert1(x_Expert1)
            x_Expert2_output = self.heads_Expert2(x_Expert2)
            x_Expert3_output = self.heads_Expert3(x_Expert3)
            x_agg_output = self.heads_agg(x_agg)

            for i in range(len(self.bn_mean_Expert2)):
                EMD_ = self.EMD(self.bn_mean_Expert2[i],self.bn_var_Expert2[i],self.in_mean_Expert2[i],self.in_var_Expert2[i])
                if i==0:
                    EMD_Expert2 = EMD_
                else:
                    EMD_Expert2 = torch.cat((EMD_Expert2,EMD_),dim=-1)    
            for i in range(len(self.bn_mean_Expert3)):
                EMD_ = self.EMD(self.bn_mean_Expert3[i],self.bn_var_Expert3[i],self.in_mean_Expert3[i],self.in_var_Expert3[i])
                if i==0:
                    EMD_Expert3 = EMD_
                else:
                    EMD_Expert3 = torch.cat((EMD_Expert3,EMD_),dim=-1)                      
            for i in range(len(self.bn_mean_Expert1)):
                EMD_ = self.EMD(self.bn_mean_Expert1[i],self.bn_var_Expert1[i],self.in_mean_Expert1[i],self.in_var_Expert1[i])
                if i==0:
                    EMD_Expert1 = EMD_
                else:
                    EMD_Expert1 = torch.cat((EMD_Expert1,EMD_),dim=-1)   
            
            n1 = EMD_Expert2.sum(-1).unsqueeze(-1)    
            n2 = EMD_Expert3.sum(-1).unsqueeze(-1)  
            n3 = EMD_Expert1.sum(-1).unsqueeze(-1)                  

            aff = F.softmax(torch.cat((n1,n2,n3),dim=-1),dim=-1)

            F_final = aff[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_Expert2+aff[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_Expert3+aff[:,2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_Expert1
            
            
            F_final_output = self.heads_F_finals(F_final)
            features_final = torch.cat((F_final_output.squeeze(),x_agg_output.squeeze()),dim=-1)

            return features_final

    def preprocess_image(self, batched_inputs):
        if self._cfg.DATASETS.NAMES[0].lower() in batched_inputs['img_paths'][0].lower():
            data_name = 'Expert1'
        elif self._cfg.DATASETS.NAMES[1].lower() in batched_inputs['img_paths'][0].lower():
            data_name = 'Expert2'
        elif self._cfg.DATASETS.NAMES[2].lower() in batched_inputs['img_paths'][0].lower():
            data_name = 'Expert3' 
        else:
            data_name = ''           
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images'].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images, data_name

    def losses(self, expert_output, F_final_output, agg_output, targets_agg, targets_expert,iters):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = agg_output['pred_class_logits'].detach()
       
        expert_outputs       = expert_output['cls_outputs']
        expert_features     = expert_output['features']
        F_final_outputs       = F_final_output['cls_outputs']
        F_final_features     = F_final_output['features']
        agg_outputs       = agg_output['cls_outputs']
        agg_features     = agg_output['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, targets_agg)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME
        
        dist_oral_ap, dist_oral_an = self.get_pair(expert_features.detach(),targets_expert)
        dist_combine_ap, dist_combine_an = self.get_pair(F_final_features,targets_expert)

        loss_dict['loss_metric_F_final'] = self.metricloss(dist_oral_ap, dist_oral_an, dist_combine_ap, dist_combine_an)
               
        if "CrossEntropyLoss" in loss_names:
            
            loss_dict['loss_cls_expert'] = cross_entropy_loss(
                expert_outputs,
                targets_expert,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

            loss_dict['loss_cls_agg'] = cross_entropy_loss(
                agg_outputs,
                targets_agg,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

        if "TripletLoss" in loss_names:
            loss_dict['loss_triplet_expert'] = triplet_loss(
                expert_features,
                targets_expert,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE

            loss_dict['loss_triplet_agg'] = triplet_loss(
                agg_features,
                targets_agg,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE
                    
        return loss_dict

    def EMD(self, u1, c1_squ, u2, c2_squ):  # u1,c1_squ [256] | u2, c2_squ [64,256]
        u1 = u1.to(self.device)
        u2 = u2.to(self.device)
        c1_squ = c1_squ.to(self.device)
        c2_squ = c2_squ.to(self.device)
        u1 = u1.repeat(u2.shape[0], 1)
        c1_squ = c1_squ.repeat(c2_squ.shape[0], 1)
        W = ((u1 - u2).pow(2).sum(dim=-1) + (c1_squ.pow(1 / 2) - c2_squ.pow(1 / 2)).pow(2).sum(dim=-1)).pow(1 / 2)
        return W.unsqueeze(-1)
                
    def init_bn_list(self):
        self.bn_mean_Expert1 = []
        self.bn_mean_Expert2 = []   
        self.bn_mean_Expert3 = []
        self.bn_var_Expert1 = []
        self.bn_var_Expert2 = []   
        self.bn_var_Expert3 = []

    def init_in_Expert1_list(self):
        self.in_mean_Expert1 = []  
        self.in_var_Expert1 = []

    def init_in_Expert2_list(self):
        self.in_mean_Expert2 = []  
        self.in_var_Expert2 = []

    def init_in_Expert3_list(self):
        self.in_mean_Expert3 = []  
        self.in_var_Expert3 = []
        
    def get_bn_list(self):
        self.init_bn_list()
        for module_name, value in self.backbone.state_dict().items():
            if 'downsample' not in module_name:
                if '1.running_mean' in module_name:
                    self.bn_mean_Expert1.append(value)
                elif '1.running_var' in module_name:
                    self.bn_var_Expert1.append(value)
                elif '2.running_mean' in module_name:
                    self.bn_mean_Expert2.append(value)
                elif '2.running_var' in module_name:
                    self.bn_var_Expert2.append(value)
                elif '3.running_mean' in module_name:
                    self.bn_mean_Expert3.append(value)
                elif '3.running_var' in module_name:
                    self.bn_var_Expert3.append(value)
                else:
                    continue
            else:
                if '2.running_mean' in module_name:
                    self.bn_mean_Expert1.append(value)
                elif '2.running_var' in module_name:
                    self.bn_var_Expert1.append(value)
                elif '3.running_mean' in module_name:
                    self.bn_mean_Expert2.append(value)
                elif '3.running_var' in module_name:
                    self.bn_var_Expert2.append(value)
                elif '4.running_mean' in module_name:
                    self.bn_mean_Expert3.append(value)
                elif '4.running_var' in module_name:
                    self.bn_var_Expert3.append(value)
                else:
                    continue
            
    def get_in_Expert1_list(self):  
        self.init_in_Expert1_list()  
        for module_name, value in self.backbone.state_dict().items():
            if 'mean_in' in module_name:
                self.in_mean_Expert1.append(value)
            elif 'var_in' in module_name:
                self.in_var_Expert1.append(value)
            else:
                continue
                
    def get_in_Expert2_list(self):  
        self.init_in_Expert2_list()  
        for module_name, value in self.backbone.state_dict().items():
            if 'mean_in' in module_name:
                self.in_mean_Expert2.append(value)
            elif 'var_in' in module_name:
                self.in_var_Expert2.append(value)
            else:
                continue
                
    def get_in_Expert3_list(self):  
        self.init_in_Expert3_list()  
        for module_name, value in self.backbone.state_dict().items():
            if 'mean_in' in module_name:
                self.in_mean_Expert3.append(value)
            elif 'var_in' in module_name:
                self.in_var_Expert3.append(value)
            else:
                continue
                
    def metricloss(self, dist_oral_ap, dist_oral_an, dist_combine_ap, dist_combine_an)-> dict:
        y = dist_oral_ap.new().resize_as_(dist_oral_ap).fill_(1)
        ap_loss = self.ranking_loss(dist_oral_ap, dist_combine_ap, y)
        an_loss = self.ranking_loss(dist_combine_an, dist_oral_an, y)
        return ap_loss+an_loss

                
    def get_pair(self, features, targets, normalize_feature=False):
        if normalize_feature:
            features = self.normalize(features, axis=-1)
            dist_mat = self.cosine_dist(features, features)
        else:
            dist_mat = self.euclidean_dist(features, features)
        N = dist_mat.size(0)

        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        dist_ap, dist_an = self.hard_example_mining(dist_mat, is_pos, is_neg)
        
        return dist_ap, dist_an  
        
    def hard_example_mining(self, dist_mat, is_pos, is_neg):
        assert len(dist_mat.size()) == 2
        N = dist_mat.size(0)
        dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)

        dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)
    
        return dist_ap, dist_an
        


    def euclidean_dist(self,x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    
    
    def cosine_dist(self,x, y):
        bs1, bs2 = x.size(0), y.size(0)
        frac_up = torch.matmul(x, y.transpose(0, 1))
        frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                    (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
        cosine = frac_up / frac_down
        return 1 - cosine
        
    def normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
          x: pytorch Variable
        Returns:
          x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x
