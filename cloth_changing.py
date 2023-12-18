import torch
import torch.nn as nn


class PoseSolider(nn.Module):
    def __init__(self, num_classes, cfg, base_weights, pose_weight):
        super(PoseSolider, self).__init__()
        self.num_classes = num_classes
        self.cfg = cfg 
        self.base = base_weights
        self.pose_weight = pose_weight
        self.in_planes = self.base.in_planes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.mha = nn.functional.scaled_dot_product_attention
        # self.mha = torch.nn.MultiheadAttention(3072, 64)
        # self.ln = nn.LayerNorm()
        self.gelu = nn.GELU()
        # self.base.cuda()
        # self.pose_weight.cuda()
        
    def forward(self, image):
        
        pose_feature, pose_represent = self.pose_weight(image)
        pose_feature_join = torch.nn.Linear(pose_feature.shape[-1], 256).cuda()(pose_feature)
        
        body_represent, body_feature, global_feature = self.base(image)
        body_feature_last_layer = global_feature[-1]
        body_feature_join = torch.nn.Linear(body_feature_last_layer.shape[-1], 256).cuda()(body_feature_last_layer)
        
        # body_feature_join = body_feature_join.transpose(1,3)
        # pose_feature_join = body_feature_join.transpose(1,3)
        
        N, D, _, _ = pose_feature_join.shape
        body_feature_join = body_feature_join.reshape(N, -1, D)
        pose_feature_join = pose_feature_join.reshape(N, -1, D) 

        out = body_feature_join + pose_feature_join
        out_lin = nn.Linear(out.shape[-1], out.shape[-1]).cuda()(out)
        out_lin = self.gelu(out_lin)
        out_lin = nn.LayerNorm(out_lin.shape[-1]).cuda()(out_lin)
        out = out + out_lin
        
        out = nn.Flatten()(out)
        # out = out.squeeze(-1)
        out = nn.Linear(out.shape[-1], self.in_planes).cuda()(out)
        score = self.classifier(out)
        print(out.shape, score.shape)
        
        return score, out