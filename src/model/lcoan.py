## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common

import torch.nn as nn
import torch
# todo 测试
# from model import kmeans_attention as attention
from model import kmeans_attention as attention


def make_model(args, parent=False):
    return LCoAN(args)


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        # modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, window_size, n_clusters, ema_decay, n_rounds):
        super(ResidualGroup, self).__init__()
        local_blocks = [
            RCAB(conv, n_feat, kernel_size, 16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]

        non_local_block = [attention.RecurrentNonLocalKmeansAttention(channels=n_feat, window_size=window_size, reduction=reduction, res_scale=res_scale, n_clusters=n_clusters, ema_decay=ema_decay, n_rounds=n_rounds)]
        m_conv_block = [conv(n_feat, n_feat, kernel_size)]

        self.local_body = nn.Sequential(*local_blocks)
        self.attention = nn.Sequential(*non_local_block)
        self.m_conv = nn.Sequential(*m_conv_block)
        self.res_scale = res_scale

    def forward(self, inputs):
        x = inputs[0]
        att_map = inputs[1]
        bucket_score = inputs[2]
        kmeans_codes = inputs[3]
        res = self.local_body(x)
        res, shared_att_map, shared_bucket_score, shared_kmeans_codes = self.attention((res, att_map, bucket_score, kmeans_codes))

        res = self.m_conv(res)
        res = res.mul(self.res_scale)
        res += x
        return res, shared_att_map, shared_bucket_score, shared_kmeans_codes


## Residual Channel Attention Network (RCAN)
class LCoAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(LCoAN, self).__init__()
        self.a = nn.Parameter(torch.Tensor([0]))
        self.a.requires_grad = True
        n_clusters = args.n_clusters
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = []
        for i in range(n_resgroups):
            m_body.append(ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale,
                                        n_resblocks=n_resblocks, window_size=args.window_size, n_clusters=n_clusters, ema_decay=args.ema_decay, n_rounds=args.n_hashes))

        m_conv_block = [conv(n_feats, n_feats, kernel_size)]

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*m_body)
        self.m_conv = nn.Sequential(*m_conv_block)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res, _, _, _ = self.body((x, None, None, None))
        res = self.m_conv(res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('msa') or name.find('a') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('msa') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
