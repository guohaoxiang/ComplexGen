# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from .attention import MultiheadAttention
from attention import MultiheadAttention


class TransformerMultipath(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="lrelu", normalize_before=False,
                 return_intermediate_dec=False, skip_encoder=False, n_path=3, num_corner_queries = 100, num_curve_queries = 100, num_patch_queries = 100, flag_decouple_pos_content = False, flag_no_tripath = False):
        super().__init__()
        #return intermediate dec set to True
        if(not skip_encoder): #not used
          encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                  dropout, activation, normalize_before)
          encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
          self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        else:
          self.encoder = None
        
        
        decoder_layer = TransformerDecoderLayerMultipath(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, flag_decouple_pos_content,flag_no_tripath = flag_no_tripath)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderMultipath(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, n_path = n_path, num_corner_queries = num_corner_queries, num_curve_queries = num_curve_queries, num_patch_queries = num_patch_queries, flag_no_tripath = flag_no_tripath)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.n_path = n_path

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed_list, pos_embed, primitive_type_embed=None,src_attention_mask=None):
        #input shape: HWxNxC
        hw, bs, c = src.shape
        query_embed_list_new = []
        for i in range(self.n_path):
            query_embed_list_new.append(query_embed_list[i].unsqueeze(1).repeat(1, bs, 1))

        tgt_list = []
        for i in range(self.n_path):
            tgt_list.append(torch.zeros_like(query_embed_list_new[i]))

        if(self.encoder is not None):
          memory = self.encoder(src, mask = src_attention_mask, src_key_padding_mask=mask, pos=pos_embed)
        else:
          memory = src

        hs_list = self.decoder(tgt_list, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos_list = query_embed_list_new,primitive_type_embed = primitive_type_embed)

        for i in range(self.n_path):
            hs_list[i] = hs_list[i].transpose(1,2)

        return hs_list, memory.permute(1, 2, 0).view(bs, c, hw)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderMultipath(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, n_path = 3, num_corner_queries = 100, num_curve_queries = 100, num_patch_queries = 100,  flag_no_tripath = False):
        super().__init__()
        layers = _get_clones(decoder_layer, num_layers) #will be released after the copy
        self.layers_list = _get_clones(layers, n_path)

        self.num_layers = num_layers
        self.norm_list = _get_clones(norm, n_path)
        self.return_intermediate = return_intermediate #true
        self.n_path = n_path
        self.num_corner_queries = num_corner_queries
        self.num_curve_queries = num_curve_queries
        self.num_patch_queries = num_patch_queries
        self.flag_no_tripath = flag_no_tripath

    def forward(self, tgt_list, memory,query_pos_list,primitive_type_embed,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                ):
        assert(len(query_pos_list) == self.n_path)
        assert(len(tgt_list) == self.n_path)
        pos_cross_list = [] #only for 3 types
        for iter1 in range(self.n_path):
            idlist = list(range(self.n_path))
            idlist.remove(iter1)
            pos_embed_list = []
            assert(len(idlist) == self.n_path - 1)
            for id in idlist:
                pos_embed_list.append(query_pos_list[id])
            pos_cross_list.append(torch.cat(pos_embed_list, dim=0))

        output_list = tgt_list
        intermediate_list = []
        for i in range(self.n_path):
            intermediate_list.append([])         

        for j in range(self.num_layers):
            #stage1
            output_selfatt_res = []
            for i in range(self.n_path):
                output_selfatt_res.append(self.layers_list[i][j](output_list[i], memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos_list[i], stage = 1))
                
                output_list[i] = output_list[i] + output_selfatt_res[i]
            
            #use normalized data to for computing
            output_normalize = []
            for i in range(self.n_path):
                output_normalize.append(self.layers_list[i][j].norm2(output_list[i]))
            if not self.flag_no_tripath:
                val_cross = [] #only for 3 types, without pritimive embedding
                # type_embed_cross = []
                key_wo_pos = []
                for iter1 in range(self.n_path):
                    idlist = list(range(self.n_path))
                    idlist.remove(iter1)
                    val_list = []
                    key_list = []
                    assert(len(idlist) == self.n_path - 1)
                    for id in idlist:
                        #output selfatt: 100,1,192
                        val_list.append(output_normalize[id])
                        key_list.append(output_normalize[id] + primitive_type_embed[id])
                    val_cross.append(torch.cat(val_list, dim=0))
                    key_wo_pos.append(torch.cat(key_list, dim=0))
            
                #stage 2 1
                output_stage2_res = []
                for i in range(self.n_path):

                    output_stage2_res.append(self.layers_list[i][j](output_normalize[i], val_cross[i], tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=None,
                                pos=pos_cross_list[i], query_pos=query_pos_list[i], stage = 2, key_wo_pos = key_wo_pos[i]))

                    output_list[i] = output_stage2_res[i] + output_list[i]

            #stage 2 2 voxel:
            output_stage2_res_voxel = []
            for i in range(self.n_path):
                output_stage2_res_voxel.append(self.layers_list[i][j](output_normalize[i], memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos_list[i], stage = 3))
                output_list[i] = output_stage2_res_voxel[i] + output_list[i]
                
            #stage 3:
            output_stage3_res = []
            for i in range(self.n_path):
                output_stage3_res.append(self.layers_list[i][j](output_list[i], memory=None, stage = 4))
                output_list[i] = output_stage3_res[i] + output_list[i]     
                if self.return_intermediate:
                    intermediate_list[i].append(self.norm_list[i](output_list[i]))
        
    
        for i in range(self.n_path):
            if self.norm_list[0] is not None:
                output_list[i] = self.norm_list[i](output_list[i])
                if self.return_intermediate:
                    intermediate_list[i].pop()
                    intermediate_list[i].append(output_list[i])

        if self.return_intermediate:
            for i in range(self.n_path):
                intermediate_list[i] = torch.stack(intermediate_list[i])
            return intermediate_list
        
        for i in range(self.n_path):
            output_list[i] = output_list[i].unsqueeze(0)
        
        return output_list


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="lrelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayerMultipath(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="lrelu", normalize_before=False, flag_decouple_pos_content = False, flag_no_tripath = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.flag_decouple_pos_content = flag_decouple_pos_content
        if not flag_decouple_pos_content:
            self.multihead_attn_voxel = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            #only impl for no depoule version
            if not flag_no_tripath:
                self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) #element
        else:
            self.multihead_attn_voxel = MultiheadAttention(2 * d_model, nhead, dropout=dropout, vdim=d_model)
            # Projection layers for voxel cross attention with split content/position
            self.voxel_ca_qcontent = nn.Linear(d_model, d_model)
            self.voxel_ca_kcontent = nn.Linear(d_model, d_model)
            self.voxel_ca_qpos_pos = nn.Linear(d_model, d_model)
            self.voxel_ca_qpos_con = nn.Linear(d_model, d_model)
            self.voxel_ca_kpos = nn.Linear(d_model, d_model)
            self.voxel_ca_v = nn.Linear(d_model, d_model)

            #multihead attn 
            self.multihead_attn_element = MultiheadAttention(2 * d_model, nhead, dropout=dropout, vdim=d_model)
            # Projection layers for element cross attention with split content/position
            self.elemt_ca_qcontent = nn.Linear(d_model, d_model)
            self.elemt_ca_kcontent = nn.Linear(d_model, d_model)
            self.elemt_ca_qpos_pos = nn.Linear(d_model, d_model)
            self.elemt_ca_qpos_con = nn.Linear(d_model, d_model)
            self.elemt_ca_kpos_pos = nn.Linear(d_model, d_model)
            self.elemt_ca_kpos_con = nn.Linear(d_model, d_model)
            self.elemt_ca_v = nn.Linear(d_model, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def with_pos_embed_decouple_pos_content(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else torch.cat([tensor, pos], dim = -1)

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos) # query pose
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward_pre_stage1(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos) # query pose
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        return tgt2

    def forward_pre_stage2_twotype(self, tgt2, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    key_wo_pos: Optional[Tensor] = None):
        if not self.flag_decouple_pos_content:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(key_wo_pos, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        else:
            q_content = self.elemt_ca_qcontent(tgt2)
            k_content = self.elemt_ca_kcontent(key_wo_pos)
            v = self.elemt_ca_v(memory)

            num_queries, bs, n_model = q_content.shape  # TODO: check shape fits our design!
            hw, _, _ = k_content.shape

            q_pos = torch.mul(self.elemt_ca_qpos_pos(query_pos), self.elemt_ca_qpos_con(tgt2))
            k_pos = torch.mul(self.elemt_ca_kpos_pos(pos), self.elemt_ca_kpos_con(key_wo_pos))

            q = q_content
            k = k_content

            q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q_pos = q_pos.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q = torch.cat([q, q_pos], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            tgt2 = self.multihead_attn_element(query=q,
                                    key=k,
                                    value=v, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0] 
        return tgt2

    def forward_pre_stage2_voxel(self, tgt2, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    key_wo_pos: Optional[Tensor] = None):
        if not self.flag_decouple_pos_content:
            tgt2 = self.multihead_attn_voxel(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        else:

            q_content = self.voxel_ca_qcontent(tgt2)
            k_content = self.voxel_ca_kcontent(memory)
            v = self.voxel_ca_v(memory)

            num_queries, bs, n_model = q_content.shape  # TODO: check shape fits our design!
            hw, _, _ = k_content.shape

            q_pos = torch.mul(self.voxel_ca_qpos_pos(query_pos), self.voxel_ca_qpos_con(tgt2))
            k_pos = self.voxel_ca_kpos(pos)

            q = q_content
            k = k_content

            q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q_pos = q_pos.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q = torch.cat([q, q_pos], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            tgt2 = self.multihead_attn_voxel(query=q,
                                    key=k,
                                    value=v, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]     

        return tgt2



    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                stage = -1,
                key_wo_pos: Optional[Tensor] = None):
        if stage == 1:
            return self.forward_pre_stage1(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        elif stage == 2:
            return self.forward_pre_stage2_twotype(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, key_wo_pos)
        elif stage == 3:
            return self.forward_pre_stage2_voxel(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, key_wo_pos)
        elif stage == 4:
            tgt = self.norm3(tgt)
            tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            return tgt
            
        
        if self.normalize_before: #true
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_transformer_tripath(args):
    return TransformerMultipath(
        d_model=args.m * 6, #32x6
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm, #true
        # return_intermediate_dec=False,
        return_intermediate_dec=not args.vis_inter_layer == -1, #modified 0127
        skip_encoder = args.skip_transformer_encoder,
        n_path = 3,
        num_corner_queries= args.num_corner_queries,
        num_curve_queries= args.num_curve_queries, 
        num_patch_queries= args.num_patch_queries, 
        flag_decouple_pos_content = False, 
        flag_no_tripath = args.no_tripath
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == 'lrelu':
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
