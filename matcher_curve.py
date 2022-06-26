# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from matcher_patch import TList
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
import numpy as np
#from pytorch3d.loss import chamfer_distance
from typing import List, Dict
TList = List[Tensor]
TDict = Dict[str, Tensor]

def curve_distance(src_points, tgt_points):
    # print('src points shape: ', src_points.shape)
    # print('tgt points shape: ', tgt_points.shape)
    distance_forward = (src_points - tgt_points).square().sum(-1).mean(-1).view(-1,1)
    distance_backward = (torch.flip(src_points, dims=(1,)) - tgt_points).square().sum(-1).mean(-1).view(-1,1)
    return torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values

def curve_distance_id(src_points, tgt_points):
    # idlist = []
    distance_forward = (src_points - tgt_points).square().sum(-1).mean(-1).view(-1,1)
    distance_backward = (torch.flip(src_points, dims=(1,)) - tgt_points).square().sum(-1).mean(-1).view(-1,1)
    
    dist = torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values
    # ids = [0] * src_points.shape[0]
    ids = torch.zeros([tgt_points.shape[0]], dtype = torch.int, device = src_points.device)
    return dist, ids


# @torch.no_grad()
# def cyclic_curve_points_66(closed_single_curve_points):
#     new_curve_points = closed_single_curve_points[:,:-1]
#     possible_curves = [new_curve_points.roll(shifts=i, dims=1) for i in range(new_curve_points.shape[1])]
#     reverse_src_points = torch.flip(new_curve_points, dims=(1,))
#     possible_curves += [reverse_src_points.roll(shifts=i, dims=1) for i in range(reverse_src_points.shape[1])]
#     possible_curves = torch.cat(possible_curves, dim=0)
#     return torch.cat([possible_curves, possible_curves[:,:1]], dim=1)

#68 points version
@torch.no_grad()
def cyclic_curve_points(closed_single_curve_points):
    new_curve_points = closed_single_curve_points[:,:]
    possible_curves = [new_curve_points.roll(shifts=i, dims=1) for i in range(new_curve_points.shape[1])]
    reverse_src_points = torch.flip(new_curve_points, dims=(1,))
    possible_curves += [reverse_src_points.roll(shifts=i, dims=1) for i in range(reverse_src_points.shape[1])]
    possible_curves = torch.cat(possible_curves, dim=0)
    # return torch.cat([possible_curves, possible_curves[:,:1]], dim=1)
    return possible_curves

def closed_curve_distance(src_points, tgt_points):
    src_points_flat = cyclic_curve_points(src_points).flatten(1,2) #from [1, 34, 3] to [66, 34, 3] to [66, -1]
    tgt_points_flat = tgt_points.flatten(1,2)
    pairwise_curve_distance = torch.cdist(src_points_flat, tgt_points_flat, p=2.0).square() / src_points.shape[1] #[66, target_number_of_curves], l2 distance normalized to single point
    return pairwise_curve_distance.min(0).values


def closed_curve_distance_id(src_points, tgt_points):
    # idlist = []
    src_points_flat = cyclic_curve_points(src_points).flatten(1,2) #from [1, 34, 3] to [66, 34, 3] to [66, -1]
    tgt_points_flat = tgt_points.flatten(1,2)
    pairwise_curve_distance = torch.cdist(src_points_flat, tgt_points_flat, p=2.0).square() / src_points.shape[1] #[66, target_number_of_curves], l2 distance normalized to single point
    ids = pairwise_curve_distance.min(0).indices
    n_half_cycle = src_points_flat.shape[0] // 2 - 1
    ids[ids > n_half_cycle] = 2 * n_half_cycle + 1 - ids[ids>n_half_cycle]
    return pairwise_curve_distance.min(0).values, ids

def chamfer_distance(src_points, tgt_points, is_src_curve_closed, flag_only_open: bool):
    if flag_only_open:
      return curve_distance(src_points, tgt_points)
    if(not is_src_curve_closed):
      return curve_distance(src_points, tgt_points)
    else:
      return closed_curve_distance(src_points, tgt_points)
    
    pairwise_distance = torch.cdist(src_points, tgt_points, p=2.0)
    #print("pairwise_distance shape=", pairwise_distance.shape)
    s2t = pairwise_distance.min(-1).values.mean(-1)
    t2s = pairwise_distance.min(-2).values.mean(-1)
    return (s2t + t2s) / 2.0

def chamfer_distance_id(src_points, tgt_points, is_src_curve_closed):
    if(not is_src_curve_closed):
      return curve_distance_id(src_points, tgt_points)
    else:
      return closed_curve_distance_id(src_points, tgt_points)

@torch.jit.script
def pairwise_shape_chamfer(src_shapes, target_shapes, gt_is_curve_closed, flag_only_open: bool):
    pairwise_distance = []
    for i in range(target_shapes.shape[0]):  #typically num_queries:100
      pairwise_distance.append(chamfer_distance(target_shapes[i].unsqueeze(0), src_shapes, gt_is_curve_closed[i], flag_only_open)) #, 
    return torch.stack(pairwise_distance).transpose(0,1)#.sqrt() #distance normalized to single point

@torch.jit.script
def pairwise_shape_chamfer_id(src_shapes, target_shapes, gt_is_curve_closed):
    pairwise_distance = []
    tgt2pred_pairid = [] #to be transpose
    for i in range(target_shapes.shape[0]):  #typically num_queries:100
      dist, ids = chamfer_distance_id(target_shapes[i].unsqueeze(0), src_shapes, gt_is_curve_closed[i])
      # pairwise_distance.append(chamfer_distance(target_shapes[i].unsqueeze(0), src_shapes, gt_is_curve_closed[i])) #, 
      pairwise_distance.append(dist)
      tgt2pred_pairid.append(ids)
    return torch.stack(pairwise_distance).transpose(0,1), torch.stack(tgt2pred_pairid).transpose(0,1)#.sqrt() #distance normalized to single point

max_cost_value = 1e6

class HungarianMatcher_Curve(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, batch_size: int, cost_class: float = 1, cost_position: float = 1, using_prob_in_matching: bool = False, flag_eval: bool = False, val_th: float = 0.5, flag_vertid: bool = True, flag_only_open: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_position = cost_position
        self.batch_size = batch_size
        self.using_prob_in_matching = using_prob_in_matching
        self.flag_eval = flag_eval
        self.val_th = val_th
        self.flag_vertid = flag_vertid
        self.flag_only_open = flag_only_open

    @torch.no_grad()
    def forward(self, outputs:TDict, target_curves_list:TList):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            target_curves_list: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "curve_points": Tensor of dim [num_target_curves, 100, 3] containing the points sampled on target curve

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        '''
        #fix the curve correpondences: which mean the first k curves of prediction and groundtruth curves are matched
        indices_copy = []
        for sample_batch_idx in range(self.batch_size):
          #print(target_curves_list[sample_batch_idx]['labels'].shape[0])
          indices_copy.append((torch.arange(target_curves_list[sample_batch_idx]['labels'].shape[0]), torch.arange(target_curves_list[sample_batch_idx]['labels'].shape[0])))
        
        return indices_copy
        '''
        bs, num_queries = outputs["pred_curve_points"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        out_valid_prob = outputs["pred_curve_logits"].softmax(-1)  # [batch_size, num_queries, 2], valid or not
        assert(len(out_valid_prob.shape) == 3 and out_valid_prob.shape[2] == 2)
        #out_valid_prob[:,:,0:1] * 
        out_type_prob = outputs["pred_curve_type"].softmax(-1)  # [batch_size, num_queries, num_classes]
        out_curve_points_position = outputs["pred_curve_points"]#.flatten(0, 1)  # [batch_size, num_queries, 100, 3]
        out_closed_prob = outputs['closed_curve_logits'].softmax(-1)
        # Also concat the target labels and boxes
        #target_curve_points_position = torch.cat([curve['curve_points'] for curve in target_curves_list]) #in shape [n, 100, 3]
        #tgt_ids = torch.cat([curve["labels"] for curve in target_curves_list])
        
        indices = []
        cycle_id = []
        for sample_batch_idx in range(self.batch_size):
          # Compute the classification cost. Contrary to the loss, we don't use the NLL,
          # but approximate it in 1 - proba[target class].
          # The 1 is a constant that doesn't change the matching, it can be ommitted.
          
          tgt_ids = target_curves_list[sample_batch_idx]['labels']
          closed_curve_gt = target_curves_list[sample_batch_idx]['is_closed']
          
          if not self.flag_eval: 
            if(not self.using_prob_in_matching):
                cost_class = - (out_type_prob[sample_batch_idx][:, tgt_ids] + 1e-6).log() - (out_valid_prob[sample_batch_idx][:, torch.zeros_like(tgt_ids)] + 1e-6).log() - (out_closed_prob[sample_batch_idx][:, closed_curve_gt] + 1e-6).log()
            else:
              cost_class = -  out_type_prob[sample_batch_idx][:, tgt_ids] -  out_valid_prob[sample_batch_idx][:, torch.zeros_like(tgt_ids)]               -  out_closed_prob[sample_batch_idx][:, closed_curve_gt]
      
            # Compute the chamfer distance between curves in batch
            # cost_curve_geometry = pairwise_shape_chamfer(out_curve_points_position[sample_batch_idx], target_curves_list[sample_batch_idx]['curve_points'].to(out_curve_points_position.device), closed_curve_gt)
            # cost_curve_geometry = pairwise_shape_chamfer(out_curve_points_position[sample_batch_idx], target_curves_list[sample_batch_idx]['curve_points'], closed_curve_gt)

            if not self.flag_vertid:
              # cost_curve_geometry = pairwise_shape_chamfer(out_curve_points_position[sample_batch_idx][valid_id], target_curves_list[sample_batch_idx]['curve_points'].to(out_curve_points_position.device), closed_curve_gt)
              cost_curve_geometry = pairwise_shape_chamfer(out_curve_points_position[sample_batch_idx], target_curves_list[sample_batch_idx]['curve_points'], closed_curve_gt, flag_only_open = self.flag_only_open)
            else:
              # cost_curve_geometry, tgt2pred_vid = pairwise_shape_chamfer_id(out_curve_points_position[sample_batch_idx][valid_id], target_curves_list[sample_batch_idx]['curve_points'].to(out_curve_points_position.device), closed_curve_gt)
              cost_curve_geometry, tgt2pred_vid = pairwise_shape_chamfer_id(out_curve_points_position[sample_batch_idx], target_curves_list[sample_batch_idx]['curve_points'], closed_curve_gt)

            # cost_curve_geometry *= target_curves_list[sample_batch_idx]['curve_length_weighting'].view(1,-1).to(cost_curve_geometry.device)
            cost_curve_geometry *= target_curves_list[sample_batch_idx]['curve_length_weighting'].view(1,-1)

            # Final cost matrix
            C = self.cost_position*cost_curve_geometry + self.cost_class * cost_class
            C = C.view(num_queries, -1).cpu()
            # indices.append(linear_sum_assignment(C))

            #repair inf error
            # C[C == np.inf] = max_cost_value

            #for debugging
            # if np.isnan(C).max():
            #   print('error sample name: ', outputs['sample_names'][sample_batch_idx])
            #   print('cost_curve_geometry: ', cost_curve_geometry)
            #   print('cost_class: ', cost_class)
            #   print('output geom info: ', out_curve_points_position[sample_batch_idx])
            #   print('gt geom info: ', target_curves_list[sample_batch_idx]['curve_points'])
            #   print('output type info: ', out_type_prob[sample_batch_idx][:, tgt_ids])
            #   print('output valid info: ', out_valid_prob[sample_batch_idx][:, torch.zeros_like(tgt_ids)])
            #   print('output close info: ', out_closed_prob[sample_batch_idx][:, closed_curve_gt])
              

            res_ass = linear_sum_assignment(C)
            indices.append(res_ass)
            # for i, j in res_ass:
            #   print(i," ",j)
            if self.flag_vertid:
              tmplist = [tgt2pred_vid[res_ass[0][i]][res_ass[1][i]] for i in range(len(res_ass[0]))]
              cycle_id.append(tmplist)
          else:
            # Compute the chamfer distance between curves in batch
            # labels = torch.argmax(out_valid_prob[sample_batch_idx], dim=-1)
            # valid_id = torch.where(labels == 0)

            valid_id = torch.where(out_valid_prob[sample_batch_idx][:,0] > self.val_th)

            # C = C.view(num_queries, -1).cpu()
            if valid_id[0].shape[0] == 0:
                tmp = np.array([], dtype=np.int64)
                indices.append((tmp,tmp))
            else:
                if not self.flag_vertid:
                  # cost_curve_geometry = pairwise_shape_chamfer(out_curve_points_position[sample_batch_idx][valid_id], target_curves_list[sample_batch_idx]['curve_points'].to(out_curve_points_position.device), closed_curve_gt)
                  cost_curve_geometry = pairwise_shape_chamfer(out_curve_points_position[sample_batch_idx][valid_id], target_curves_list[sample_batch_idx]['curve_points'], closed_curve_gt, flag_only_open = self.flag_only_open)
                else:
                  # cost_curve_geometry, tgt2pred_vid = pairwise_shape_chamfer_id(out_curve_points_position[sample_batch_idx][valid_id], target_curves_list[sample_batch_idx]['curve_points'].to(out_curve_points_position.device), closed_curve_gt)
                  cost_curve_geometry, tgt2pred_vid = pairwise_shape_chamfer_id(out_curve_points_position[sample_batch_idx][valid_id], target_curves_list[sample_batch_idx]['curve_points'], closed_curve_gt)

                # cost_curve_geometry *= target_curves_list[sample_batch_idx]['curve_length_weighting'].view(1,-1).to(cost_curve_geometry.device)
                cost_curve_geometry *= target_curves_list[sample_batch_idx]['curve_length_weighting'].view(1,-1)

                # Final cost matrix
                C = self.cost_position*cost_curve_geometry
                C = C.view(valid_id[0].shape[0], -1).cpu()
                (pred_id, tar_id) = linear_sum_assignment(C)
                pred_id = valid_id[0][pred_id]
                indices.append((pred_id, tar_id))

            # C = C.view(valid_id[0].shape[0], -1).cpu()
            # (pred_id, tar_id) = linear_sum_assignment(C)
            # pred_id = valid_id[0][pred_id]
            # indices.append((pred_id, tar_id))
        
        if not self.flag_vertid:
          return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
          return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], [torch.stack(i) for i in cycle_id]


def build_matcher_curve(args, flag_eval = False):
  if not flag_eval:
    return HungarianMatcher_Curve(args.batch_size, cost_class=args.class_loss_coef, cost_position=args.curve_geometry_loss_coef, using_prob_in_matching=args.using_prob_in_matching, flag_vertid=args.flag_cycleid, flag_only_open = args.curve_open_loss)#bs, 1,1000, False
  else:
    return HungarianMatcher_Curve(args.batch_size, cost_class=0.0, cost_position=1, using_prob_in_matching=False, flag_eval = True, val_th = args.val_th, flag_vertid=args.flag_cycleid, flag_only_open = args.curve_open_loss)#bs, 1,1000, False
    # return HungarianMatcher_Curve(args.batch_size, cost_class=0.0, cost_position=1, using_prob_in_matching=False, flag_eval = True, val_th = args.val_th, flag_vertid=True)#bs, 1,1000, False

# def build_matcher_curve(args, flag_eval = False):
#   if not flag_eval:
#     return torch.jit.script(HungarianMatcher_Curve(args.batch_size, cost_class=args.class_loss_coef, cost_position=args.curve_geometry_loss_coef, using_prob_in_matching=args.using_prob_in_matching))#bs, 1,1000, False
#   else:
#     return torch.jit.script(HungarianMatcher_Curve(args.batch_size, cost_class=0.0, cost_position=1, using_prob_in_matching=False, flag_eval = True, val_th = args.val_th))#bs, 1,1000, False
#     # return HungarianMatcher_Curve(args.batch_size, cost_class=0.0, cost_position=1, using_prob_in_matching=False, flag_eval = True, val_th = args.val_th, flag_vertid=True)#bs, 1,1000, False
