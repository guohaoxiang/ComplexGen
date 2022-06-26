# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
from typing import List
from chamferdist import knn_points
#from pytorch3d.loss import chamfer_distance

#  = 10

TList = List[Tensor]
def chamfer_distance_patch(src_points, tgt_points, single_dir_patch_chamfer: bool, flag_batch_cd : bool):
    if not flag_batch_cd:    
      pairwise_distance = torch.cdist(src_points, tgt_points, p=2.0).square()
      #print("pairwise_distance shape=", pairwise_distance.shape)
      # assert(pairwise_distance.shape[-1] == 100) #pairwise_distance.shape = [n_prediction, target_points_num, 100(10*10)]
      assert(pairwise_distance.shape[0] > 0) #pairwise_distance.shape = [n_prediction, target_points_num, 100(10*10)]
      s2t = pairwise_distance.min(-1).values.mean(-1)
      if(single_dir_patch_chamfer):
        return s2t
      t2s = pairwise_distance.min(-2).values.mean(-1)
      return (s2t + 0.2*t2s) / 1.2
    else:
      #knn version
      num_queries = tgt_points.shape[0]
      src_points_batch = src_points.repeat(num_queries, 1,1)
      src_nn = knn_points(src_points_batch, tgt_points)
      s2t_batch = src_nn.dists[...,0].mean(-1)
      if (single_dir_patch_chamfer):
        return s2t_batch
      target_nn = knn_points(tgt_points, src_points_batch)
      t2s_batch = target_nn.dists[...,0].mean(-1)
      return (s2t_batch + 0.2*t2s_batch) / 1.2
  # else:
  #   #simulated emd
  #   #rot 1 in 8 tgt_pts here
  #   res = []
  #   for ids in emd_idlist:
  #     res.append((tgt_points[:,ids,:] - src_points).square().sum(-1).mean(-1).unsqueeze(0))
  #   res = torch.cat(res, 0)
  #   return res.min(0).values

# @torch.jit.script  #not available for flag_batch_cd
def pairwise_shape_chamfer_patch(src_shapes, target_shapes: TList, single_dir_patch_chamfer: bool, flag_batch_cd: bool = False):
    pairwise_distance = []
    # for i in range(len(target_shapes)):  #typically num_queries:100
    #   # pairwise_distance.append(chamfer_distance_patch(target_shapes[i].unsqueeze(0).to(src_shapes.device), src_shapes, single_dir_patch_chamfer)) 
    #   pairwise_distance.append(chamfer_distance_patch(target_shapes[i].unsqueeze(0), src_shapes, single_dir_patch_chamfer))

    ll = torch.tensor([len(t) for t in target_shapes])
    assert(ll.min() > 0)
    for item in target_shapes:  #typically num_queries:100
      # pairwise_distance.append(chamfer_distance_patch(target_shapes[i].unsqueeze(0).to(src_shapes.device), src_shapes, single_dir_patch_chamfer)) 
      pairwise_distance.append(chamfer_distance_patch(item.unsqueeze(0), src_shapes, single_dir_patch_chamfer, flag_batch_cd)) 

    return torch.stack(pairwise_distance).transpose(0,1) #distance normalized to single point

@torch.jit.script
def emd_by_id(gt: Tensor, pred: Tensor, gtid: Tensor, points_per_patch_dim: int):
  #gt shape: N/1 100, 3
  #pred shape: N, 100, 3
  #return N,
  # print('gt shape: ', gt.shape)
  # print('pred shape: ', pred.shape)
  gt_batch = gt[:, gtid, :].view(len(gt), -1, points_per_patch_dim * points_per_patch_dim, 3)
  pred_batch = pred.view(len(pred), 1, points_per_patch_dim * points_per_patch_dim, 3)
  dist = (gt_batch - pred_batch).square().sum(-1).mean(-1).min(-1).values
  return dist

class HungarianMatcher_Patch(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, batch_size, cost_class: float = 1, cost_position: float = 1, using_prob_in_matching=False, single_dir_patch_chamfer=False, flag_batch_cd = False, flag_patch_emd = False, flag_eval=False, val_th = 0.3, flag_patch_uv = False, dim_grid = 10):
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
        self.single_dir_patch_chamfer = single_dir_patch_chamfer
        self.flag_eval = flag_eval
        self.flag_batch_cd = flag_batch_cd
        self.flag_patch_uv = flag_patch_uv
        self.dim_grid = dim_grid
        self.val_th = val_th
        self.emd_idlist = []
        #fill only if flag_patch_emd = true
        self.flag_patch_emd = False
        if flag_patch_emd:
          self.flag_patch_emd = True
          base = torch.arange(dim_grid * dim_grid).view(dim_grid, dim_grid)
          # self.emd_idlist.append(base.flatten())
          for i in range(4):
            self.emd_idlist.append(torch.rot90(base, i, [0,1]).flatten())
          
          base_t = base.transpose(0,1)
          # self.emd_idlist.append(base_t.flatten())
          for i in range(4):
            self.emd_idlist.append(torch.rot90(base_t, i, [0,1]).flatten())
          self.emd_idlist = torch.cat(self.emd_idlist) #800
        
        if flag_patch_uv:
          self.emd_idlist_u = []
          self.emd_idlist_v = []
          base = torch.arange(dim_grid * dim_grid).view(dim_grid,  dim_grid)
          #set idlist u
          for i in range(dim_grid):
            cur_base = base.roll(shifts=i, dims = 0)
            for i in range(0,4,2):
              self.emd_idlist_u.append(torch.rot90(cur_base, i, [0,1]).flatten())
            
            cur_base = cur_base.transpose(0,1)
            for i in range(1,4,2):
              self.emd_idlist_u.append(torch.rot90(cur_base, i, [0,1]).flatten())
          
          self.emd_idlist_u = torch.cat(self.emd_idlist_u)
          # #set idlist v
          # for i in range(points_per_patch_dim):
          #   cur_base = base.roll(shifts=i, dims = 1)
          #   for i in range(4):
          #     self.emd_idlist_v.append(torch.rot90(cur_base, i, [0,1]).flatten())
            
          #   cur_base = cur_base.transpose(0,1)
          #   for i in range(4):
          #     self.emd_idlist_v.append(torch.rot90(cur_base, i, [0,1]).flatten())
          
          # self.emd_idlist_v = torch.cat(self.emd_idlist_v)

    #compute distance inside

    def emd(self, src_points, tgt_points, uclosed, vclosed):
      #src is gt here
      if not self.flag_patch_uv:
        # return self.emd_open(src_points, tgt_points)
        return emd_by_id(src_points, tgt_points, self.emd_idlist, self.dim_grid)
      if uclosed:
        return emd_by_id(src_points, tgt_points, self.emd_idlist_u, self.dim_grid)
      # if vclosed:
      #   return emd_by_id(src_points, tgt_points, self.emd_idlist_v, points_per_patch_dim)

      return emd_by_id(src_points, tgt_points, self.emd_idlist, self.dim_grid)
      
    def pairwise_shape_emd(self, src_shapes, target_shapes, target_uclosed, target_vclosed):
      #assume that either u closed or v closed
      pairwise_distance = []
      assert(len(target_shapes) == len(target_uclosed) and len(target_shapes) == len(target_vclosed))
      # for i in range(len(src_shapes)):
      #   print('i: {} src shapes: {}'.format(i, src_shapes[i].shape))

      for i in range(len(target_shapes)):
        # print('i: {} target shapes: {}'.format(i, target_shapes[i].shape))
        pairwise_distance.append(self.emd(target_shapes[i].unsqueeze(0), src_shapes, target_uclosed[i], target_vclosed[i])) 
      
        

      return torch.stack(pairwise_distance).transpose(0,1) #distance normalized to single point
    
    @torch.no_grad()
    def forward(self, outputs, target_patches_list):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            target_patches_list: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
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
        bs, num_queries = outputs["pred_patch_points"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_valid_prob = outputs["pred_patch_logits"].softmax(-1)  # [batch_size, num_queries, 2]
        assert(len(out_valid_prob.shape) == 3 and out_valid_prob.shape[2] == 2)
        #out_valid_prob[:,:,0:1] * 
        out_type_prob = outputs["pred_patch_type"].softmax(-1)  # [batch_size, num_queries, num_classes]
        out_patch_points_position = outputs["pred_patch_points"]#.flatten(0, 1)  # [batch_size, num_queries, 100, 3]
        # Also concat the target labels and boxes
        #target_curve_points_position = torch.cat([curve['curve_points'] for curve in target_patches_list]) #in shape [n, 100, 3]
        #tgt_ids = torch.cat([curve["labels"] for curve in target_patches_list])
        
        indices = []
        for sample_batch_idx in range(len(target_patches_list)):
          # Compute the classification cost. Contrary to the loss, we don't use the NLL,
          # but approximate it in 1 - proba[target class].
          # The 1 is a constant that doesn't change the matching, it can be ommitted.
          
          tgt_ids = target_patches_list[sample_batch_idx]['labels']
          #closed_curve_gt = target_patches_list[sample_batch_idx]['is_closed']
          if self.flag_patch_emd:
            uclosed_patch_gt = target_patches_list[sample_batch_idx]['u_closed']
            vclosed_patch_gt = target_patches_list[sample_batch_idx]['v_closed']
          if not self.flag_eval:
            if(not self.using_prob_in_matching):
              cost_class = - (out_type_prob[sample_batch_idx][:, tgt_ids] + 1e-6).log() - (out_valid_prob[sample_batch_idx][:, torch.zeros_like(tgt_ids)] + 1e-6).log()
            else:
              cost_class = -  out_type_prob[sample_batch_idx][:, tgt_ids]                - out_valid_prob[sample_batch_idx][:, torch.zeros_like(tgt_ids)]
    
            # Compute the chamfer distance between curves in batch
            if not self.flag_patch_emd:
              cost_patch_geometry = pairwise_shape_chamfer_patch(out_patch_points_position[sample_batch_idx], target_patches_list[sample_batch_idx]['patch_points'], self.single_dir_patch_chamfer, self.flag_batch_cd)
            else:
              cost_patch_geometry = self.pairwise_shape_emd(out_patch_points_position[sample_batch_idx], target_patches_list[sample_batch_idx]['patch_points'], uclosed_patch_gt, vclosed_patch_gt)
            
            # cost_patch_geometry *= target_patches_list[sample_batch_idx]['patch_area_weighting'].view(1, -1).to(cost_patch_geometry.device)
            cost_patch_geometry *= target_patches_list[sample_batch_idx]['patch_area_weighting'].view(1, -1)
            
            # Final cost matrix
            C = self.cost_position*cost_patch_geometry + self.cost_class * cost_class
            C = C.view(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
          else:
            # labels = torch.argmax(out_valid_prob[sample_batch_idx], dim=-1)
            # valid_id = torch.where(labels == 0)
            valid_id = torch.where(out_valid_prob[sample_batch_idx][:,0] > self.val_th)
            # print('output valid prob shape: ', out_valid_prob[sample_batch_idx].shape)
            # print('valid id shape: ', valid_id[0].shape)
            # print('output valid prob: ', out_valid_prob[sample_batch_idx])
            # print('valid shape: ', valid_id[0].shape)
            if valid_id[0].shape[0] == 0:
              continue
            if not self.flag_patch_emd:
              cost_patch_geometry = pairwise_shape_chamfer_patch(out_patch_points_position[sample_batch_idx][valid_id], target_patches_list[sample_batch_idx]['patch_points'], self.single_dir_patch_chamfer , self.flag_batch_cd)
            else:
              cost_patch_geometry = self.pairwise_shape_emd(out_patch_points_position[sample_batch_idx][valid_id], target_patches_list[sample_batch_idx]['patch_points'], uclosed_patch_gt, vclosed_patch_gt)
            
            # cost_patch_geometry *= target_patches_list[sample_batch_idx]['patch_area_weighting'].view(1, -1).to(cost_patch_geometry.device)
            cost_patch_geometry *= target_patches_list[sample_batch_idx]['patch_area_weighting'].view(1, -1)

            
            # Final cost matrix
            C = self.cost_position*cost_patch_geometry
            if valid_id[0].shape[0] == 0:
                tmp = np.array([], dtype=np.int64)
                indices.append((tmp,tmp))
            else:
                # print('C shape: ', C.shape)
                # print('valid shape: ', valid_id[0].shape)
                C = C.view(valid_id[0].shape[0], -1).cpu()
                (pred_id, tar_id) = linear_sum_assignment(C)
                pred_id = valid_id[0][pred_id]
                indices.append((pred_id, tar_id))
            # C = C.view(valid_id[0].shape[0], -1).cpu()
            # (pred_id, tar_id) = linear_sum_assignment(C)
            # pred_id = valid_id[0][pred_id]
            # indices.append((pred_id, tar_id))

            # C = C.view(num_queries, -1).cpu()
            # indices.append(linear_sum_assignment(C))
        
        if len(indices) != 0:
        # if True:
          return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
          return []


def build_matcher_patch(args, flag_eval = False):
  if not flag_eval:
    return HungarianMatcher_Patch(args.batch_size, cost_class=args.class_loss_coef, cost_position=args.patch_geometry_loss_coef, using_prob_in_matching=args.using_prob_in_matching, single_dir_patch_chamfer=args.single_dir_patch_chamfer, flag_batch_cd=args.batch_cd, flag_patch_emd = args.patch_emd, flag_patch_uv=args.patch_uv, dim_grid = args.points_per_patch_dim)
  else:
    return HungarianMatcher_Patch(args.batch_size, cost_class=0.0, cost_position=1.0, using_prob_in_matching=False, single_dir_patch_chamfer=False,flag_batch_cd=args.batch_cd, flag_eval = True, val_th = args.val_th, flag_patch_emd = args.patch_emd, flag_patch_uv=args.patch_uv, dim_grid = args.points_per_patch_dim)
