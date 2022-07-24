# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np


class HungarianMatcher_Corner(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_position: float = 1, using_prob_in_matching=False, flag_eval = False, val_th = 0.5):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_position = cost_position
        self.using_prob_in_matching = using_prob_in_matching
        self.flag_eval = flag_eval
        self.val_th = val_th #only used for evaluation

    @torch.no_grad()
    def forward(self, outputs, target_corner_position_list):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_corner_position"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_corner_position = outputs["pred_corner_position"].flatten(0, 1)  # [batch_size * num_queries, 3]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if not self.flag_eval:
            target_corner_position = torch.cat([pos for pos in target_corner_position_list])
            
            #certainly, target corner points should be all non-empty
            tgt_ids = torch.zeros(target_corner_position.shape[0], dtype=torch.long)
            if(not self.using_prob_in_matching):
                cost_class = -(out_prob[:, tgt_ids] + 1e-6).log()
            else:
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L2 cost between corners
            cost_corner_position = torch.cdist(out_corner_position, target_corner_position, p=2).square()

            # Final cost matrix
            C = self.cost_position*cost_corner_position + self.cost_class * cost_class
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(target_corner_position_entry) for target_corner_position_entry in target_corner_position_list]
            #for i, c in enumerate(C.split(sizes, -1)): print(c[i].shape, c[i])
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
            out_corner_position = out_corner_position.view(bs, num_queries, 3)
            out_prob = out_prob.view(bs,num_queries, 2)
            indices = []
            for sample_batch_idx in range(bs):
                valid_id = torch.where(out_prob[sample_batch_idx][:,0] > self.val_th)
                cost_corner_position = torch.cdist(out_corner_position[sample_batch_idx][valid_id], target_corner_position_list[sample_batch_idx], p=2).square()
                C = self.cost_position*cost_corner_position
                if valid_id[0].shape[0] == 0:
                    tmp = np.array([], dtype=np.int64)
                    indices.append((tmp,tmp))
                else:
                    C = C.view(valid_id[0].shape[0], -1).cpu()
                    (pred_id, tar_id) = linear_sum_assignment(C)
                    pred_id = valid_id[0][pred_id]
                    indices.append((pred_id, tar_id))
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher_corner(args, flag_eval = False):
    if not flag_eval:
        return HungarianMatcher_Corner(cost_class=args.class_loss_coef, cost_position=args.corner_geometry_loss_coef, using_prob_in_matching=args.using_prob_in_matching)
    else:
        return HungarianMatcher_Corner(cost_class=0.0, cost_position=1.0, using_prob_in_matching=False, flag_eval = True, val_th = args.val_th)
