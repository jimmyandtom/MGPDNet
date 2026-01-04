import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
import matplotlib.pyplot as plt
from .attention import MultiHeadAttention
from .attention import MultiLayerPerceptron
from .attention import SELayer


class PrototypeDiversityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(PrototypeDiversityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, diverse_protos):
        B, K, C = diverse_protos.shape
        if K <= 1:
            return torch.tensor(0.0, device=diverse_protos.device)

        protos_norm = F.normalize(diverse_protos, p=2, dim=-1)
        sim_matrix = torch.bmm(protos_norm, protos_norm.transpose(1, 2))
        
        loss = torch.abs(sim_matrix)
        loss = loss - torch.eye(K, device=sim_matrix.device).unsqueeze(0)
        
        loss = torch.sum(loss, dim=[1, 2]) / (K * (K - 1))
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# --- FPS ---
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# ================= SELayer =================

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (N, C)
        """
        # Squeeze: (N, C) -> (1, N, C) -> (1, C, N)
        b, n, c = 1, x.shape[0], x.shape[1] 
        x_unsqueeze = x.unsqueeze(0).transpose(1, 2) #  (1, C, N) 
        y = self.avg_pool(x_unsqueeze).squeeze(-1) # -> (1, C)
        
        # Excitation
        y = self.fc(y) # -> (1, C)
        
        # Fusing: (N, C) * (C) -> (N, C)
        return x * y.squeeze(0)

# ================= GAT =================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

        self.a = nn.Linear(2 * out_features, 1, bias=False)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h shape: (N, in_features), adj shape: (N, N)
        Wh = self.W(h) # (N, out_features)
        
        N = Wh.size(0)
        Wh1 = Wh.unsqueeze(1).expand(N, N, -1)
        Wh2 = Wh.unsqueeze(0).expand(N, N, -1)
        a_input = torch.cat([Wh1, Wh2], dim=-1)
        e = self.leakyrelu(self.a(a_input).squeeze(2))
        

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = attention * adj
        
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
class GraphAwarePrototypeGenerator(nn.Module):
    def __init__(self, num_archetypes, num_contextuals_per_archetype, in_channels, criterion_diversity, 
                 k_neighbors=10, max_graph_nodes=512, top_nodes_per_archetype=32):
        super().__init__()
        self.K = num_archetypes
        self.M_per_K = num_contextuals_per_archetype
        self.M = self.K * self.M_per_K
        self.top_n = top_nodes_per_archetype
        
        self.in_channels = in_channels
        self.k_neighbors = k_neighbors
        self.max_graph_nodes = max_graph_nodes
        self.criterion_diversity = criterion_diversity

        # --- Canonical queries ---
        self.canonical_queries = nn.Parameter(torch.randn(self.K, in_channels))
        self.query_identity_embedding = nn.Embedding(self.K, in_channels)

        # --- Appearance queries ---
        # Only M_per_K sets of queries are required, as they will be reused across the K regions
        self.appearance_queries = nn.Parameter(torch.randn(self.M_per_K, in_channels))
        
        self.gat_layer = GraphAttentionLayer(in_channels, in_channels, dropout=0.1, concat=False)
        self.se_layer = SELayer(in_channels, reduction=4)
        
    def forward(self, support_features, support_mask, global_prototypes):

        B, C, H, W = support_features.shape
        device = support_features.device
        diversity_loss = torch.tensor(0.0, device=device)
        mask_i = F.interpolate(support_mask.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=True).squeeze() > 0.5
        fg_features_i = support_features[0, :, mask_i]
        
        if fg_features_i.shape[1] < self.k_neighbors:
            # Fallback logic
            pooled_canon = global_prototypes[0].unsqueeze(0).repeat(self.K, 1)
            pooled_app = global_prototypes[0].unsqueeze(0).repeat(self.M, 1)
            return pooled_canon.unsqueeze(0), pooled_app.unsqueeze(0), diversity_loss

        graph_nodes = fg_features_i.t()
        if graph_nodes.shape[0] > self.max_graph_nodes:
            fps_indices = farthest_point_sample(graph_nodes.unsqueeze(0), self.max_graph_nodes).squeeze(0)
            graph_nodes = graph_nodes[fps_indices]

        with torch.no_grad():
            nodes_norm_sim = F.normalize(graph_nodes, p=2, dim=-1)
            sim_matrix = torch.mm(nodes_norm_sim, nodes_norm_sim.t())
        k = min(self.k_neighbors, graph_nodes.shape[0])
        _, topk_indices = torch.topk(sim_matrix, k, dim=1)
        adj_binary = torch.zeros_like(sim_matrix).scatter_(1, topk_indices, 1)
        adj_weighted = (adj_binary + adj_binary.t()).gt(0).float() * sim_matrix
        refined_nodes = self.se_layer(graph_nodes + self.gat_layer(graph_nodes, adj_weighted))
        
        # --- Canonical Prototypes ---
        canon_queries_norm = F.normalize(self.canonical_queries, p=2, dim=-1)
        nodes_norm = F.normalize(refined_nodes, p=2, dim=-1)
        affinity_canon = torch.matmul(canon_queries_norm, nodes_norm.t())
        canonical_prototypes = torch.matmul(F.softmax(affinity_canon, dim=1), refined_nodes)
        
        if self.training:
            query_indices = torch.arange(self.K, device=device)
            identity_vectors = self.query_identity_embedding(query_indices)
            specialized_queries = self.canonical_queries + identity_vectors
            diversity_loss = self.criterion_diversity(specialized_queries.unsqueeze(0))

        # --- Appearance Prototypes ---
        all_appearance_prototypes = []
        with torch.no_grad():
            # affinity_canon: [K, N_nodes]
            _, top_node_indices_for_each_canon = torch.topk(affinity_canon, k=min(self.top_n, refined_nodes.shape[0]), dim=1)

        for i in range(self.K):
            # 1. Retrieve the nodes within the influence range of the i-th paradigm prototype
            local_node_indices = top_node_indices_for_each_canon[i]
            local_nodes = refined_nodes[local_node_indices] # [top_n, C]
            
            # 2. "Within this local region, extract M_per_K apparent prototypes
            app_queries_norm = F.normalize(self.appearance_queries, p=2, dim=-1)
            local_nodes_norm = F.normalize(local_nodes, p=2, dim=-1)
            
            # [M_per_K, top_n]
            affinity_app_local = torch.matmul(app_queries_norm, local_nodes_norm.t())
            # [M_per_K, C]
            appearance_prototypes_local = torch.matmul(F.softmax(affinity_app_local, dim=1), local_nodes)
            
            all_appearance_prototypes.append(appearance_prototypes_local)
        
        appearance_prototypes = torch.cat(all_appearance_prototypes, dim=0)

        return canonical_prototypes.unsqueeze(0), appearance_prototypes.unsqueeze(0), diversity_loss

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class HybridGraphRefiner(nn.Module):
    def __init__(self, in_channels, hid_channels, drop_path_rate=0.1):
        super(HybridGraphRefiner, self).__init__()
        
        # --- Core 1: Guided Self-Attention (for Structure) ---
        self.norm_self1 = nn.LayerNorm(in_channels)
        self.qkv_self = nn.Conv1d(in_channels, in_channels * 2 + in_channels, 1) # Q, K, V
        self.proj_self = nn.Conv1d(in_channels, in_channels, 1)
        self.semantic_gate_generator = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())
        
        # --- Core 2: Purified Cross-Attention (for Appearance) ---
        self.norm_cross1 = nn.LayerNorm(in_channels)
        self.q_proj_cross = nn.Conv1d(in_channels, hid_channels, 1)
        self.proto_kv_proj_cross = nn.Linear(in_channels, hid_channels * 2)
        self.out_proj_cross = nn.Conv1d(hid_channels, in_channels, 1)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # --- Fusion Module ---
        self.fusion_block = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, qry_fts, archetypes, contextuals):
        B, C, H, W = qry_fts.shape
        qry_fts_flat = qry_fts.view(B, C, -1)

        # --- Structure Stream ---
        structural_guide = torch.mean(archetypes, dim=1) # [B, C]
        
        x1_norm = self.norm_self1(qry_fts_flat.transpose(1, 2)).transpose(1, 2)
        q_self, k_self, v_self = self.qkv_self(x1_norm).chunk(3, dim=1)
        
        semantic_gate = self.semantic_gate_generator(structural_guide).unsqueeze(-1)
        k_self_guided = k_self * semantic_gate
        
        attn_self = F.softmax(torch.bmm(q_self.transpose(1,2), k_self_guided) / (k_self.size(1)**0.5), dim=-1)
        refined_self = self.proj_self(torch.bmm(v_self, attn_self.transpose(1,2)))
        output_self = qry_fts_flat + self.drop_path(refined_self)

        # --- Appearance Stream ---
        x2_norm = self.norm_cross1(qry_fts_flat.transpose(1, 2)).transpose(1, 2)
        q_cross = self.q_proj_cross(x2_norm)

        k_cross, v_cross = self.proto_kv_proj_cross(contextuals).chunk(2, dim=-1)
        k_cross, v_cross = k_cross.transpose(1,2), v_cross.transpose(1,2)
        
        attn_cross = F.softmax(torch.bmm(q_cross.transpose(1,2), k_cross) / (k_cross.size(1)**0.5), dim=-1)
        refined_cross = self.out_proj_cross(torch.bmm(v_cross, attn_cross.transpose(1,2)))
        output_cross = qry_fts_flat + self.drop_path(refined_cross)
        
        # --- Fusion ---
        output_self = output_self.view(B, C, H, W)
        output_cross = output_cross.view(B, C, H, W)
        fused_input = torch.cat([output_self, output_cross], dim=1)
        fused_output = self.fusion_block(fused_input)

        return qry_fts + fused_output

class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss() 
        self.criterion_MSE = nn.MSELoss()
        self.alpha = torch.Tensor([1.0, 0.])
        self.encoder_out_channels = [512, 512] 
        self.iter = 3
        self.fg_sampler = np.random.RandomState(1289)
        self.fg_num = 64  # number of foreground partitions
        # Add an adaptive pooling layer to resize to fg_num
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.fg_num)

        self.criterion_diversity = PrototypeDiversityLoss()
        self.prototype_generator = GraphAwarePrototypeGenerator(
            num_archetypes=8,
            num_contextuals_per_archetype=7,
            in_channels=512,
            criterion_diversity=self.criterion_diversity,
            k_neighbors=10,
            max_graph_nodes=512,
        )
        
        self.query_refiner = HybridGraphRefiner(in_channels=512, hid_channels=256)
        self.proj_low = nn.Conv2d(256, 512, kernel_size=1, bias=False)
    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False, t_loss_scaler=1, n_iters=30):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            qry_mask: query mask
                1 x H x W  tensor
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, low_level_fts_dict, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        supp_fts_low_level = [low_level_fts_dict[dic][:(self.n_ways * self.n_shots * supp_bs)].view(
            supp_bs, self.n_ways, self.n_shots, -1, *low_level_fts_dict[dic].shape[-2:]) for _, dic in enumerate(low_level_fts_dict)]
        qry_fts_low_level = [low_level_fts_dict[dic][(self.n_ways * self.n_shots * supp_bs):].view(
            qry_bs, self.n_queries, -1, *low_level_fts_dict[dic].shape[-2:]) for _, dic in enumerate(low_level_fts_dict)]

        
        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]  # t for support features
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        # Compute loss #
        align_loss = torch.zeros(1).to(self.device)
        loss_qry = torch.zeros(1).to(self.device) 
        contrastive_loss = torch.zeros(1).to(self.device)
        outputs = []
        refined_qry_fts = []
        for epi in range(supp_bs):

            # Extract prototypes #
            supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                         range(len(supp_fts))]
            fg_prototypes = [self.getPrototype(supp_fts_[n]) for n in range(len(supp_fts))]  # prototype for support

            temp_diversity_loss = torch.tensor(0.0, device=self.device)
            archetypes_list = []
            contextuals_list = []
            for n in range(len(supp_fts)):
                for way in range(self.n_ways):
                    for shot in range(self.n_shots):
                        archetypes, contextuals, div_loss = self.prototype_generator(
                            supp_fts[n][[epi], way, shot], 
                            supp_mask[[epi], way, shot], 
                            fg_prototypes[n][way]
                        )
                        if train:
                            temp_diversity_loss += div_loss
                archetypes_list.append(archetypes)
                contextuals_list.append(contextuals)
            
            if train:
                num_calls = len(supp_fts) * self.n_ways * self.n_shots
                contrastive_loss += (temp_diversity_loss / num_calls)
            
            qry_pred = [torch.stack(
                [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'  
            qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], qry_pred[n][epi]) for n in range(len(qry_fts))]

            # Compute loss use query coarse prototype
            if train:
                qry_pred = [torch.stack(
                    [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
                qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], qry_pred[n][epi]) for n in
                                        range(len(qry_fts))]

                qry_pred = [self.getPred(qry_fts[n][epi], qry_prototype_coarse[n], self.thresh_pred[epi])
                            for n in range(len(qry_fts))]  # N x Wa x H' x W'
                # qry_pred[0]: (1, 32, 32)  qry_pred[1]: (1, 16, 16)
                qry_pred = [F.interpolate(qry_pred[n][None, ...], size=img_size, mode='bilinear', align_corners=True)
                            for n in range(len(qry_fts))]
                preds = [self.alpha[n] * qry_pred[n] for n in range(len(qry_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                preds = torch.cat((1.0 - preds, preds), dim=1)

                qry_label = torch.full_like(qry_mask[epi], 255, device=qry_mask.device)
                qry_label[qry_mask[epi] == 1] = 1
                qry_label[qry_mask[epi] == 0] = 0
                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                loss_qry += self.criterion(log_prob, qry_label[None, ...].long()) / self.n_shots / self.n_ways
                #qry_loss 

            for n in range(len(supp_fts)):    
                current_qry_fts = qry_fts[n][epi]
                # --- Fine-grained Enhancement with Diverse Prototypes ---
               
                refined_fts_stage2 = self.query_refiner(
                                                qry_fts=current_qry_fts,
                                                archetypes=archetypes_list[n],
                                                contextuals=contextuals_list[n]
                                            )
       
                refined_qry_fts.append(refined_fts_stage2)

            high_level_features = refined_qry_fts[0] # (B, 512, H/16, W/16)
            low_level_features = qry_fts_low_level[0][epi] # (B, 256, H/4, W/4)
            low_level_projected = self.proj_low(low_level_features)
            high_level_upsampled = F.interpolate(high_level_features, 
                                                 size=low_level_projected.shape[-2:], 
                                                 mode='bilinear', 
                                                 align_corners=True)
            

            fused_features_for_pred = high_level_upsampled + low_level_projected
            supp_proto_for_pred = fg_prototypes[0][0]
            qry_pred_supp = self.getPred(fused_features_for_pred, supp_proto_for_pred, self.thresh_pred[0])
            qry_pred_up_supp = F.interpolate(qry_pred_supp.unsqueeze(1), size=img_size, mode='bilinear', align_corners=True)
            
            preds = qry_pred_up_supp
            preds = torch.cat((1.0 - preds, preds), dim=1)
            outputs.append(preds)
            
            ################################################################################

            # Prototype alignment loss #
            if train:
           
                align_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                [refined_qry_fts[n] for n in range(len(qry_fts))],
                                                preds, supp_mask[epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / supp_bs, loss_qry / supp_bs, 0.02 * contrastive_loss

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

       
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
   
        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
    
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'
   
        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred_[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)

                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))



                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

