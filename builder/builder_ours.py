"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, topk, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.topk = topk

        print('MoCo with auxiliary loss (TopK={})'.format(self.topk))

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # create the queue_backbone
        self.register_buffer("queue_backbone", torch.randn(dim_mlp, K))
        self.queue_backbone = nn.functional.normalize(self.queue_backbone, dim=0)

        # hook for getting intermediate feature
        self.fhook_q  = getattr(self.encoder_q, 'avgpool').register_forward_hook(self.forward_hook_q_())
        self.fhook_k = getattr(self.encoder_k, 'avgpool').register_forward_hook(self.forward_hook_k_())

    def forward_hook_q_(self):
        def hook(module, input, output):
            self.features_q = torch.flatten(output, 1)
        return hook

    def forward_hook_k_(self):
        def hook(module, input, output):
            self.features_k = torch.flatten(output, 1)
        return hook

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_backbone):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        keys_backbone = concat_all_gather(keys_backbone)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_backbone[:, ptr:ptr + batch_size] = keys_backbone.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def pull_topk(self, logits_backbone, logits_out):
        # input:
        #   logits_backbone: B x K 
        #   logits_out: B x K
        # (1) Find nearest neighbors in the memory bank after the avgpool layer.
        # (2) Maximize agreement between query and its nearest neighbors after MLP,
        # while minimizing the agreement with other keys.

        # Find knn
        knn = torch.topk(logits_backbone, dim=1, k=self.topk)[1]
        
        # Mask -> all zeros, with ones for the nearest neighbors
        mask = torch.scatter(torch.zeros_like(logits_out), 1, knn, 1)        

        # Relax with temperature
        logits_out /= self.T
        
        # For numerical stability
        logits_max, _ = torch.max(logits_out, dim=1, keepdim=True)
        logits_out = logits_out - logits_max.detach()

        # compute log prob (based on SupContrast)
        exp_logits_out = torch.exp(logits_out)
        log_prob_out = logits_out - torch.log(exp_logits_out.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob_out).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos.mean()

        return loss

    def forward(self, im_q, im_k, im_q_small=None, svm=False):
        """
        Input:
            im_q: a batch of query images (bs, 3, h, w)
            im_k: a batch of key images (bs, 3, h, w)
            im_q_small: a batch of key images (bs*4, 3, h_small, w_small)
        Output:
            logits, targets, pull-loss
        """
        if svm:
            return self.encoder_q(im_q)

        b = im_q.shape[0]

        # compute query features
        q = self.encoder_q(im_q)  # queries: bs x dim
        q = nn.functional.normalize(q, dim=-1)
        feat_dim = q.shape[-1]
        q = q.unsqueeze(1) # bs x 1 x dim

        # compute backbone features
        q_backbone = nn.functional.normalize(self.features_q, dim=-1)
        feat_dim_backbone = q_backbone.shape[-1]
        q_backbone = q_backbone.unsqueeze(1)
        
        if im_q_small is not None:
            q_small = self.encoder_q(im_q_small).view(b, -1, feat_dim) # bs x 4 x dim
            q_small = nn.functional.normalize(q_small, dim=-1)
            q = torch.cat((q, q_small), dim=1) # bs x 5 x dim
            
            # compute backbone features
            q_small_backbone = nn.functional.normalize(self.features_q, dim=-1)
            q_small_backbone = q_small_backbone.view(b, -1, feat_dim_backbone)
            q_backbone = torch.cat((q_backbone, q_small_backbone), dim=1) # bs x 5 x dim

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: bs x dim
            k = nn.functional.normalize(k, dim=1)
            k_backbone = nn.functional.normalize(self.features_k, dim=-1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_backbone = self._batch_unshuffle_ddp(k_backbone, idx_unshuffle)
            
        # compute logits
        # positive logits
        l_pos = (q * k.unsqueeze(1)).sum(2).view(-1, 1) 

        # negative logits
        l_neg = torch.einsum('nc,ck->nk', [q.view(-1, feat_dim), self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # enqueue
        self._dequeue_and_enqueue(k[:b], k_backbone[:b])
        
        # negative logits backbone
        l_neg_backbone = torch.einsum('nc,ck->nk', [q_backbone.view(-1, feat_dim_backbone), self.queue_backbone.clone().detach()])

        return logits, labels, self.pull_topk(l_neg_backbone, l_neg)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
