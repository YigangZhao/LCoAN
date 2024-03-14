import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
from inspect import isfunction
from operator import mul
from functools import reduce, wraps
import math


class Kmeans(nn.Module):
    def __init__(self, n_rounds, qk_dim, n_clusters, ema_decay=0.999):
        super().__init__()
        self.n_rounds = n_rounds
        self.n_clusters = n_clusters
        self.ema_decay = ema_decay
        # todo LSH rotated vectors [N, n_hashes, H*W, hash_buckets]
        self.register_buffer('means', torch.randn(n_rounds, n_clusters, qk_dim))
        self.register_buffer('initted', torch.tensor(False))
        self.num_new_means = 0
        self.new_means = None

    @torch.no_grad()
    def init(self, x):
        # todo 理解x的shape [batch, n_rounds, length, dim]
        if self.initted:
            return
        _, h, _, d, device, dtype = *x.shape, x.device, x.dtype

        n_clusters = self.means.shape[1]

        # 一个min batch内的所有特征聚合在一起
        means = x.transpose(0, 1).contiguous().view(h, -1, d)
        # 一个min batch内所有特征的数量
        n_samples = means.shape[1]

        if n_samples >= n_clusters:
            indices = torch.randperm(n_samples, device=device)[:n_clusters]
        else:
            indices = torch.randint(0, n_samples, (n_clusters,), device=device)

        means = means[:, indices]

        for _ in range(KMEANS_INIT_ITERS):
            # todo kmeans更新迭代函数，暂时可以不用细究
            means = kmeans_iter(x, means)

        self.num_new_means = 0
        self.means.data.copy_(means)
        self.initted.data.copy_(torch.tensor(True))

    # todo 这里num_new_means置零了，需要关注
    # @torch.no_grad()
    # def update(self, new_means=None):
    #     new_means = default(new_means, self.new_means)
    #     assert exists(new_means), 'new kmeans has not been supplied'
    #     ema_inplace(self.means, new_means, self.ema_decay)
    #
    #     del self.new_means
    #     self.new_means = None
    #     self.num_new_means = 0

    def forward(self, x, update_means=False):
        x = expand_dim(x, 1, self.n_rounds)
        self.init(x)

        b, dtype = x.shape[0], x.dtype
        means = self.means.type(dtype)
        x = F.normalize(x, 2, dim=-1).type(dtype)

        with torch.no_grad():
            dists, buckets = dists_and_buckets(x, means)

        # todo 暂时不考虑损失函数
        # routed_means = batched_index_select(expand_dim(means, 0, b), buckets)
        # loss = F.mse_loss(x, routed_means) * self.commitment  # commitment损失贡献比例

        if update_means:
            with torch.no_grad():
                means = kmeans_iter(x, means, buckets)
            # todo 这里控制kmeans更新速度，消融实验测试结果
            if self.ema_decay <= 0:
                ema_decay = self.num_new_means / (self.num_new_means + 1)
            else:
                ema_decay = self.ema_decay
            self.new_means = ema(self.new_means, means, ema_decay)
            # self.new_means = ema(self.new_means, means, )
            self.num_new_means += 1

        # add offsets to avoid bucket codes overlapping between multi-round kmeans
        offsets = torch.arange(self.n_rounds, device=x.device)
        offsets = torch.reshape(offsets * self.n_clusters, (1, -1, 1))
        bucket_codes = torch.reshape(buckets + offsets, (b, -1,))  # [N,n_hashes*H*W]

        # return dists, loss
        return bucket_codes  # [batch, n_rounds, length] LSH [N,n_hashes*H*W]


class NonLocalKmeansAttention(nn.Module):
    def __init__(self, channels=256, k_size=3, reduction=4, n_clusters=128, window_size=144,
                 conv=common.default_conv, res_scale=0.1, ema_decay=0.999, n_rounds=1):
        super(NonLocalKmeansAttention, self).__init__()
        self.window_size = window_size
        self.n_rounds = n_rounds
        self.reduction = reduction
        self.res_scale = res_scale
        self.kmeans = Kmeans(n_rounds, channels // reduction, n_clusters, ema_decay)
        self.conv_match = common.BasicBlock(conv, channels, channels // reduction, k_size, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(conv, channels, channels, 1, bn=False, act=None)

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, input_x):
        N, _, H, W = input_x.shape

        x_embed = self.conv_match(input_x).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        y_embed = self.conv_assembly(input_x).view(N, -1, H * W).contiguous().permute(0, 2, 1)

        L, C = x_embed.shape[-2:]

        kmeans_codes = self.kmeans(x_embed, self.training)
        kmeans_codes = kmeans_codes.detach()

        # group elements with same hash code by sorting
        # 一开始, 不同的n_hash, kmeans_code数值范围不同，直接排序也可以区分开[0: H * W - 1], [H * W, H * W * 2 - 1], ...
        _, indices = kmeans_codes.sort(dim=-1)  # [N,n_hashes*H*W]
        _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
        mod_indices = (indices % L)  # now range from (0->H*W)

        # x_embed 扩展了 n_hash
        x_embed_sorted = common.batched_index_select(x_embed, mod_indices)  # [N,n_hashes*H*W,C]
        y_embed_sorted = common.batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C]

        # pad the embedding if it cannot be divided by window_size
        padding = self.window_size - L % self.window_size if L % self.window_size != 0 else 0
        # 按照n_hashes调整形状
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_rounds, -1, C))  # [N, n_hashes, H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_rounds, -1, C * self.reduction))

        if padding:
            pad_x = x_att_buckets[:, :, -padding:, :].clone()
            pad_y = y_att_buckets[:, :, -padding:, :].clone()
            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

        # 按照window_size调整形状
        x_att_buckets = torch.reshape(x_att_buckets, (
        N, self.n_rounds, -1, self.window_size, C))  # [N, n_hashes, num_windows, window_size, C]
        y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_rounds, -1, self.window_size, C * self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)

        # allow attend to adjacent buckets
        # 将前后的桶连接[当前桶，后一个桶，前一个桶]
        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        # unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets,
                                 x_match)  # [N, n_hashes, num_windows, window_size, window_size*3]

        # softmax
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)  # (after softmax) reuse
        bucket_score = torch.reshape(bucket_score, [N, self.n_rounds, -1])


        # attention
        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets)  # [N, n_hashes, num_windows, window_size, C]
        ret = torch.reshape(ret, (N, self.n_rounds, -1, C * self.reduction))

        # if padded, then remove extra elements
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        # recover the original order
        ret = torch.reshape(ret, (N, -1, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,))  # [N,n_hashes*H*W]
        ret = common.batched_index_select(ret, undo_sort)  # [N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N,n_hashes*H*W]

        # weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_rounds, L, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_rounds, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input_x
        return ret


class RecurrentNonLocalKmeansAttention(nn.Module):
    def __init__(self, channels=256, k_size=3, reduction=4, n_clusters=128, window_size=144,
                 conv=common.default_conv, res_scale=0.1, ema_decay=0.999, n_rounds=1):
        super(RecurrentNonLocalKmeansAttention, self).__init__()
        self.window_size = window_size
        self.n_rounds = n_rounds
        self.reduction = reduction
        self.res_scale = res_scale
        self.kmeans = Kmeans(n_rounds, channels // reduction, n_clusters, ema_decay)
        self.conv_match = common.BasicBlock(conv, channels, channels // reduction, k_size, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(conv, channels, channels, 1, bn=False, act=None)

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, inputs):
        input_x = inputs[0]
        N, _, H, W = input_x.shape
        shared_att_map = inputs[1]
        shared_bucket_score = inputs[2]
        shared_kmeans_codes = inputs[3]

        x_embed = self.conv_match(input_x).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        y_embed = self.conv_assembly(input_x).view(N, -1, H * W).contiguous().permute(0, 2, 1)

        L, C = x_embed.shape[-2:]

        if shared_att_map is None:
            kmeans_codes = self.kmeans(x_embed, self.training)
            kmeans_codes = kmeans_codes.detach()

            # group elements with same hash code by sorting
            # 一开始, 不同的n_hash, kmeans_code数值范围不同，直接排序也可以区分开[0: H * W - 1], [H * W, H * W * 2 - 1], ...
            _, indices = kmeans_codes.sort(dim=-1)  # [N,n_hashes*H*W]
            _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
            mod_indices = (indices % L)  # now range from (0->H*W)

            # x_embed 扩展了 n_hash
            x_embed_sorted = common.batched_index_select(x_embed, mod_indices)  # [N,n_hashes*H*W,C]
            y_embed_sorted = common.batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C]

            # pad the embedding if it cannot be divided by window_size
            padding = self.window_size - L % self.window_size if L % self.window_size != 0 else 0
            # 按照n_hashes调整形状
            x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_rounds, -1, C))  # [N, n_hashes, H*W,C]
            y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_rounds, -1, C * self.reduction))

            if padding:
                pad_x = x_att_buckets[:, :, -padding:, :].clone()
                pad_y = y_att_buckets[:, :, -padding:, :].clone()
                x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
                y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

            # 按照window_size调整形状
            x_att_buckets = torch.reshape(x_att_buckets, (N, self.n_rounds, -1, self.window_size, C))  # [N, n_hashes, num_windows, window_size, C]
            y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_rounds, -1, self.window_size, C * self.reduction))

            x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)

            # allow attend to adjacent buckets
            # 将前后的桶连接[当前桶，后一个桶，前一个桶]
            x_match = self.add_adjacent_buckets(x_match)
            y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

            # unormalized attention score
            raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets, x_match)  # [N, n_hashes, num_windows, window_size, window_size*3]

            # softmax
            bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
            score = torch.exp(raw_score - bucket_score)  # (after softmax) reuse
            bucket_score = torch.reshape(bucket_score, [N, self.n_rounds, -1])

            shared_kmeans_codes = kmeans_codes
            shared_att_map = score
            shared_bucket_score = bucket_score
        else:
            kmeans_codes = shared_kmeans_codes.detach()

            # group elements with same hash code by sorting
            # 一开始, 不同的n_hash, kmeans_code数值范围不同，直接排序也可以区分开[0: H * W - 1], [H * W, H * W * 2 - 1], ...
            _, indices = kmeans_codes.sort(dim=-1)  # [N,n_hashes*H*W]
            _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
            mod_indices = (indices % L)  # now range from (0->H*W)

            y_embed_sorted = common.batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C]

            # pad the embedding if it cannot be divided by window_size
            padding = self.window_size - L % self.window_size if L % self.window_size != 0 else 0

            # 按照n_hashes调整形状
            y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_rounds, -1, C * self.reduction))

            if padding:
                pad_y = y_att_buckets[:, :, -padding:, :].clone()
                y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

            # 安装window_size调整形状
            y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_rounds, -1, self.window_size, C * self.reduction))

            y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

            score = shared_att_map
            bucket_score = shared_bucket_score

        # attention
        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets)  # [N, n_hashes, num_windows, window_size, C]
        ret = torch.reshape(ret, (N, self.n_rounds, -1, C * self.reduction))

        # if padded, then remove extra elements
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        # recover the original order
        ret = torch.reshape(ret, (N, -1, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,))  # [N,n_hashes*H*W]
        ret = common.batched_index_select(ret, undo_sort)  # [N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N,n_hashes*H*W]

        # weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_rounds, L, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_rounds, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input_x
        return ret, shared_att_map, shared_bucket_score, shared_kmeans_codes


class RecurrentNonLocalKmeansAttentionWithLoss(nn.Module):
    def __init__(self, channels=256, k_size=3, reduction=4, n_clusters=128, window_size=144,
                 conv=common.default_conv, res_scale=0.1, ema_decay=0.999, n_rounds=1):
        super(RecurrentNonLocalKmeansAttentionWithLoss, self).__init__()
        self.window_size = window_size
        self.n_rounds = n_rounds
        self.reduction = reduction
        self.res_scale = res_scale
        self.kmeans = Kmeans(n_rounds, channels // reduction, n_clusters, ema_decay)
        self.conv_match = common.BasicBlock(conv, channels, channels // reduction, 3, bn=False, act=None)
        self.conv_match2 = common.BasicBlock(conv, channels, channels // reduction, 3, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(conv, channels, channels, 1, bn=False, act=None)

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, inputs):
        input_x = inputs[0]
        N, _, H, W = input_x.shape
        shared_att_map = inputs[1]
        shared_bucket_score = inputs[2]
        shared_kmeans_codes = inputs[3]

        x_embed_1 = self.conv_match(input_x)

        x_embed = x_embed_1.view(N, -1, H * W).contiguous().permute(0, 2, 1)
        y_embed = self.conv_assembly(input_x).view(N, -1, H * W).contiguous().permute(0, 2, 1)

        L, C = x_embed.shape[-2:]

        loss = 0
        if self.training:
            k = math.sqrt(6)
            x_embed_2 = self.conv_match2(input_x)
            x_embed_2 = F.normalize(x_embed_2, p=2, dim=1, eps=5e-5) * k
            x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5) * k
            score = torch.matmul(x_embed_1.permute(0, 2, 3, 1).view((N, H * W, -1)),
                                 x_embed_2.view(N, -1, H * W))  # [N,H*W,H*W]
            score = torch.exp(score)
            score = torch.sort(score, dim=2, descending=True)[0]
            positive = torch.mean(score[:, :, :15], dim=2)
            negative = torch.mean(score[:, :, 50:65], dim=2)  # [N,H*W]
            loss = F.relu(-1 * torch.log(positive / (negative + 1e-6))+1)
            loss = torch.mean(loss)


        if shared_att_map is None:
            kmeans_codes = self.kmeans(x_embed, self.training)
            kmeans_codes = kmeans_codes.detach()

            # group elements with same hash code by sorting
            # 一开始, 不同的n_hash, kmeans_code数值范围不同，直接排序也可以区分开[0: H * W - 1], [H * W, H * W * 2 - 1], ...
            _, indices = kmeans_codes.sort(dim=-1)  # [N,n_hashes*H*W]
            _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
            mod_indices = (indices % L)  # now range from (0->H*W)

            # x_embed 扩展了 n_hash
            x_embed_sorted = common.batched_index_select(x_embed, mod_indices)  # [N,n_hashes*H*W,C]
            y_embed_sorted = common.batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C]

            # pad the embedding if it cannot be divided by window_size
            padding = self.window_size - L % self.window_size if L % self.window_size != 0 else 0
            # 按照n_hashes调整形状
            x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_rounds, -1, C))  # [N, n_hashes, H*W,C]
            y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_rounds, -1, C * self.reduction))

            if padding:
                pad_x = x_att_buckets[:, :, -padding:, :].clone()
                pad_y = y_att_buckets[:, :, -padding:, :].clone()
                x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
                y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

            # 按照window_size调整形状
            x_att_buckets = torch.reshape(x_att_buckets, (N, self.n_rounds, -1, self.window_size, C))  # [N, n_hashes, num_windows, window_size, C]
            y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_rounds, -1, self.window_size, C * self.reduction))

            x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)

            # allow attend to adjacent buckets
            # 将前后的桶连接[当前桶，后一个桶，前一个桶]
            x_match = self.add_adjacent_buckets(x_match)
            y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

            # unormalized attention score
            raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets, x_match)  # [N, n_hashes, num_windows, window_size, window_size*3]

            # softmax
            bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
            score = torch.exp(raw_score - bucket_score)  # (after softmax) reuse
            bucket_score = torch.reshape(bucket_score, [N, self.n_rounds, -1])

            shared_kmeans_codes = kmeans_codes
            shared_att_map = score
            shared_bucket_score = bucket_score
        else:
            kmeans_codes = shared_kmeans_codes.detach()

            # group elements with same hash code by sorting
            # 一开始, 不同的n_hash, kmeans_code数值范围不同，直接排序也可以区分开[0: H * W - 1], [H * W, H * W * 2 - 1], ...
            _, indices = kmeans_codes.sort(dim=-1)  # [N,n_hashes*H*W]
            _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
            mod_indices = (indices % L)  # now range from (0->H*W)

            y_embed_sorted = common.batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C]

            # pad the embedding if it cannot be divided by window_size
            padding = self.window_size - L % self.window_size if L % self.window_size != 0 else 0

            # 按照n_hashes调整形状
            y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_rounds, -1, C * self.reduction))

            if padding:
                pad_y = y_att_buckets[:, :, -padding:, :].clone()
                y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

            # 安装window_size调整形状
            y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_rounds, -1, self.window_size, C * self.reduction))

            y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

            score = shared_att_map
            bucket_score = shared_bucket_score

        # attention
        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets)  # [N, n_hashes, num_windows, window_size, C]
        ret = torch.reshape(ret, (N, self.n_rounds, -1, C * self.reduction))

        # if padded, then remove extra elements
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        # recover the original order
        ret = torch.reshape(ret, (N, -1, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,))  # [N,n_hashes*H*W]
        ret = common.batched_index_select(ret, undo_sort)  # [N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N,n_hashes*H*W]

        # weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_rounds, L, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_rounds, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input_x
        return ret, shared_att_map, shared_bucket_score, shared_kmeans_codes, loss

# constants

TOKEN_SELF_ATTN_VALUE = -5e4
KMEANS_INIT_ITERS = 10


# helper functions

def exists(val):
    return val is not None


def identity(x, *args, **kwargs):
    return x


def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x


def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if exists(cache):
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def compose(*fns):
    def inner(x, *args, **kwargs):
        for fn in reversed(fns):
            x = fn(x, *args, **kwargs)
        return x

    return inner


def to(t):
    return {'device': t.device, 'dtype': t.dtype}


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


def is_empty(t):
    return t.nelement() == 0


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    ed = expand_dim(indices, -1, last_dim)
    return values.gather(2, ed)


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def scatter_mean(src, t, index, dim, eps=1e-5):
    numer = src.scatter_add(dim, index, t)
    denom = src.scatter_add(dim, index, torch.ones_like(t))
    return numer / (denom + eps)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim + 1] = split_dims
    return t.reshape(shape)


def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


# helper classes
def map_first_tuple_or_el(x, fn):
    if isinstance(x, tuple):
        return (fn(x[0]),) + x[1:]
    return fn(x)


# kmeans related function and class
# def update_kmeans_on_backwards(module):
#     module.kmean_modules = find_modules(module, Kmeans)
#
#     def hook(_, grad_in, grad_out):
#         for m in module.kmean_modules:
#             m.update()
#
#     return module.register_backward_hook(hook)  # 在指定网络层执行完backward（）之后调用钩子函数


def similarity(x, means):
    return torch.einsum('bhld,hcd->bhlc', x, means)


def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


def batched_bincount(index, num_classes, dim=-1):
    # [16, 1, 2304] -> [16, 1, 24] 统计每个桶内有多少特征
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def kmeans_iter(x, means, buckets=None):
    b, h, l, d, dtype, num_clusters = *x.shape, x.dtype, means.shape[1]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_clusters).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    # 初始化质心向量，置零
    means_ = buckets.new_zeros(b, h, num_clusters, d, dtype=dtype)
    # todo 好像是所有的向量相加为质心的向量
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)

    # torch.where(condition, a, b)其中输入参数condition: 条件限制，如果满足条件，则选择a；否则选择b作为输出。
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means


def distribution(dists, window_size):
    _, topk_indices = dists.topk(k=window_size, dim=-2)
    indices = topk_indices.transpose(-2, -1)
    return indices.reshape(*indices.size()[:2], -1)

