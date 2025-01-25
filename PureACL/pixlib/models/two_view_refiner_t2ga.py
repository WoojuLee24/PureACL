"""
The top-level model of training-time sidfm.
Encapsulates the feature extraction, pose optimization, loss and metrics.
"""
import torch
from torch.nn import functional as nnF
import logging
from copy import deepcopy
import omegaconf
import itertools

from PureACL.pixlib.models.base_model import BaseModel
from PureACL.pixlib.models import get_model
from PureACL.pixlib.models.utils import masked_mean, merge_confidence_map, extract_keypoints, camera_to_onground
from PureACL.pixlib.geometry.losses import scaled_barron
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

pose_loss = True

def get_weight_from_reproloss(err):
    # the reprojection loss is from 0 to 16.67 ,tensor[B]
    err = err.detach()
    weight = torch.ones_like(err)*err
    weight[err < 10.] = 0
    weight = torch.clamp(weight, min=0., max=50.)
    return weight

class TwoViewRefiner(BaseModel):
    default_conf = {
        'extractor': {
            'name': 'unet', #'s2dnet',
        },
        'optimizer': {
            'name': 'learned_optimizer', #'basic_optimizer',
        },
        'duplicate_optimizer_per_scale': False,
        'success_thresh': 3,
        'clamp_error': 50,
        'normalize_features': True,
        'normalize_dt': True,
        'debug': False,
        'topk': 256,
        # deprecated entries
        'init_target_offset': None,

        'grd_height': 1.65,
    }
    required_data_keys = {
        'ref': ['image', 'camera', 'T_w2cam'],
        'query': ['image', 'camera', 'T_w2cam'],
    }
    strict_conf = False  # need to pass new confs to children models

    def _init(self, conf):
        self.extractor = get_model(conf.extractor.name)(conf.extractor)
        assert hasattr(self.extractor, 'scales')

        Opt = get_model(conf.optimizer.name)
        if conf.duplicate_optimizer_per_scale:
            oconfs = [deepcopy(conf.optimizer) for _ in self.extractor.scales]
            feature_dim = self.extractor.conf.output_dim
            if not isinstance(feature_dim, int):
                for d, oconf in zip(feature_dim, oconfs):
                    with omegaconf.read_write(oconf):
                        with omegaconf.open_dict(oconf):
                            oconf.feature_dim = d
            self.optimizer = torch.nn.ModuleList([Opt(c) for c in oconfs])
        else:
            self.optimizer = Opt(conf.optimizer)

        if conf.init_target_offset is not None:
            raise ValueError('This entry has been deprecated. Please instead '
                             'use the `init_pose` config of the dataloader.')
        self.grd_height = conf.grd_height

    def add_mlp(self):
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.mlp0 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
        )
        self.mlp0 = self.mlp0.cuda()
        self.mlp1 = self.mlp1.cuda()
        self.mlp2 = self.mlp2.cuda()


    def _forward(self, data):
        def process_siamese(data_i, data_type):
            if data_type == 'ref':
                data_i['type'] = 'sat'
                data_i['q2r'] = data['T_q2r_init']
            else:
                data_i['type'] = 'grd'
                data_i['normal'] = data['normal']
            pred_i = self.extractor(data_i)
            pred_i['camera_pyr'] = [data_i['camera'].scale(1 / s)
                                    for s in self.extractor.scales]
            return pred_i
        # start_time = time.time()
        if 'query_3' in data.keys():
            pred = {i: process_siamese(data[i], i) for i in ['ref', 'query', 'query_1', 'query_2','query_3']}
        elif 'query_1' in data.keys():
            pred = {i: process_siamese(data[i], i) for i in ['ref', 'query', 'query_1']}
        else:
            pred = {i: process_siamese(data[i], i) for i in ['ref', 'query']}
        # after_time = time.time()
        # print('duration:',after_time-start_time)

        confidence_count = 2
        if pred['ref']['confidences'][0].size(1) == 1:
            confidence_count = 1

        # find ground key points from confidence map. top from each grd_img
        if 'query_3' in data.keys():
            query_list = ['query', 'query_1', 'query_2', 'query_3']
        elif 'query_1' in data.keys():
            query_list = ['query', 'query_1']
        else:
            query_list = ['query']
        for q in query_list:
            # find 2d key points from grd confidence map
            grd_key_confidence = merge_confidence_map(pred[q]['confidences'],confidence_count) #[B,H,W]
            p2d_grd_key = extract_keypoints(grd_key_confidence, topk=self.conf.topk, start_ratio = data[q]['camera'].c[0,1]/data[q]['camera'].size[0,1]+0.05) #data['grd_ratio'][0])

            # turn grd key points from 2d to 3d, assume points are on ground
            p2d_c_key = data[q]['camera'].image2world(p2d_grd_key) # 2D->3D scale unknown
            p3d_grd_key = camera_to_onground(p2d_c_key, data[q]['T_w2cam'], data[q]['camera_h'], data['normal'])

            p2d_grd_key_new = expand_y_coordinates(p2d_grd_key, 432)
            # turn grd key points from 2d to 3d, assume points are on ground
            p2d_c_key_new = data[q]['camera'].image2world(p2d_grd_key_new)  # 2D->3D scale unknown
            p3d_grd_key_new = camera_to_onground(p2d_c_key_new, data[q]['T_w2cam'], data[q]['camera_h'], data['normal'])

            if q == 'query':
                p3D_query = p3d_grd_key
                p3D_query_new = p3d_grd_key_new
            else:
                p3D_query = torch.cat([p3D_query, p3d_grd_key], dim=1)
                p3D_query_new = torch.cat([p3D_query_new, p3d_grd_key_new], dim=1)
        pred['query']['grd_key_3d'] = p3D_query

        T_init = data['T_q2r_init']

        pred['T_q2r_init'] = []
        pred['T_q2r_opt'] = []
        pred['T_q2r_opt_list'] = []
        pred['pose_loss'] = []
        for i in reversed(range(len(self.extractor.scales))):
            F_ref = pred['ref']['feature_maps'][i]
            cam_ref = pred['ref']['camera_pyr'][i]

            if self.conf.duplicate_optimizer_per_scale:
                opt = self.optimizer[i]
            else:
                opt = self.optimizer

            if 'query_3' in data.keys():
                querys = ['query', 'query_1', 'query_2', 'query_3']
            elif 'query_1' in data.keys():
                querys = ['query', 'query_1']
            else:
                querys = ['query']

            if i == 2:
                mlp = self.mlp2
            elif i == 1:
                mlp = self.mlp1
            elif i == 0:
                mlp = self.mlp0

            W_q = None
            F_q = None
            mask = None
            for q in querys:
                F_q_cur = pred[q]['feature_maps'][i]
                cam_q = pred[q]['camera_pyr'][i]
t2ga
                p2D_query, visible = cam_q.world2image(data[q]['T_w2cam'] * p3D_query)
                F_q_cur, mask_cur, _ = opt.interpolator(F_q_cur, p2D_query)
                mask_cur &= visible

                p2D_query_new, visible_new = cam_q.world2image(data[q]['T_w2cam'] * p3D_query_new)
                F_q_cur_value, mask_cur_new, _ = opt.interpolator(pred[q]['feature_maps'][i], p2D_query_new)
                mask_cur_new &= visible_new


                F_q_cur_key_new = mlp(F_q_cur_value)
                mask = create_mask(p2D_query, p2D_query_new, 432)

                F_q_cur, weights = scaled_dot_product_attention(F_q_cur, F_q_cur_key_new, F_q_cur_value, mask)


                W_q_cur = pred[q]['confidences'][i]
                W_q_cur, _, _ = opt.interpolator(W_q_cur, p2D_query)
                # merge W_q_cur to W_q
                if W_q is None:
                    W_q = W_q_cur * mask_cur[:,:,None]
                else:
                    # check repeat
                    multi_projection = torch.logical_and(mask, mask_cur)
                    if W_q_cur.size(2) > 1:
                        reset = W_q_cur[:,:,0]*W_q_cur[:,:,1] * multi_projection > W_q[:,:,0]*W_q[:,:,1] * multi_projection
                    else:
                        reset = W_q_cur[:, :, 0] * multi_projection > W_q[:, :, 0] * multi_projection
                    mask = mask & (~reset)
                    mask_cur = mask_cur & ~(multi_projection & ~reset)

                    W_q = W_q_cur * mask_cur[:,:,None] + W_q * mask[:,:,None]

                if F_q is None:
                    F_q = F_q_cur * mask_cur[:,:,None]
                    mask = mask_cur
                else:
                    F_q = F_q_cur * mask_cur[:,:,None] + F_q * mask[:,:,None]
                    mask = torch.logical_or(mask, mask_cur)



            W_ref = pred['ref']['confidences'][i]
            W_ref_q = (W_ref, W_q, confidence_count)

            if self.conf.normalize_features:
                F_q = nnF.normalize(F_q, dim=2)  # B x N x C
                F_ref = nnF.normalize(F_ref, dim=1)  # B x C x W x H

            T_opt, failed, T_opt_list = opt(dict(
                p3D=p3D_query, F_ref=F_ref, F_q=F_q, T_init=T_init, camera=cam_ref,
                mask=mask, W_ref_q=W_ref_q))

            pred['T_q2r_init'].append(T_init)
            pred['T_q2r_opt'].append(T_opt)
            pred['T_q2r_opt_list'].append(T_opt_list)

            T_init = T_opt.detach()

            # add by shan, query & reprojection GT error, for query unet back propogate
            if pose_loss:
                loss_gt = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_gt'], cam_ref, mask=mask, W_ref_query=W_ref_q)
                loss_init = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_init'], cam_ref, mask=mask, W_ref_query=W_ref_q)
                diff_loss = torch.log(1 + torch.exp(10*(1- (loss_init + 1e-8) / (loss_gt + 1e-8))))
                pred['pose_loss'].append(diff_loss)

        return pred

    def preject_l1loss(self, opt, p3D, F_ref, F_query, T_gt, camera, mask=None, W_ref_query= None):
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        res, valid, w_unc, _, _ = opt.cost_fn.residuals(T_gt, *args)
        if mask is not None:
            valid &= mask

        # compute the cost and aggregate the weights
        cost = (res ** 2).sum(-1)
        cost, w_loss, _ = opt.loss_fn(cost) # robust cost
        loss = cost * valid.float()
        if w_unc is not None:
            loss = loss * w_unc

        return torch.sum(loss, dim=-1)/(torch.sum(valid)+1e-6)

    def loss(self, pred, data):
        cam_ref = data['ref']['camera']
        points_3d = pred['query']['grd_key_3d']
        def project(T_q2r):
            return cam_ref.world2image(T_q2r * points_3d)

        p2D_r_gt, mask = project(data['T_q2r_gt'])
        p2D_r_i, mask_i = project(data['T_q2r_init'])
        mask = (mask & mask_i).float()

        def reprojection_error(T_q2r):
            p2D_r, _ = project(T_q2r)
            err = torch.sum((p2D_r_gt - p2D_r)**2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0]/4
            err = masked_mean(err, mask, -1)
            return err

        err_init = reprojection_error(pred['T_q2r_init'][0])

        num_scales = len(self.extractor.scales)
        success = None
        losses = {'total': 0.}
        if pose_loss:
            losses['pose_loss'] = 0
        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            if i > 0:
                loss = loss * success.float()
            thresh = self.conf.success_thresh * self.extractor.scales[-1-i]
            success = err < thresh
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss

            # add by shan, query & reprojection GT error, for query unet back propogate
            if pose_loss:
                losses['pose_loss'] += pred['pose_loss'][i]/ num_scales
                # poss_loss_weight = 5
                poss_loss_weight = get_weight_from_reproloss(err_init)
                losses['total'] += (poss_loss_weight * pred['pose_loss'][i]/ num_scales).clamp(max=self.conf.clamp_error/num_scales)

        losses['reprojection_error'] = err
        losses['reprojection_error/init'] = err_init

        return losses

    # def metrics(self, pred, data):
    #     T_r2q_gt = data['T_q2r_gt'].inv()
    #
    #     @torch.no_grad()
    #     def scaled_pose_error(T_q2r):
    #         err_R, err_t = (T_r2q_gt@T_q2r).magnitude()
    #         err_lat, err_long = (T_r2q_gt@T_q2r).magnitude_latlong()
    #         return err_R, err_t, err_lat, err_long
    #
    #     metrics = {}
    #     for i, T_opt in enumerate(pred['T_q2r_opt']):
    #         err = scaled_pose_error(T_opt)
    #         metrics[f'R_error/{i}'], metrics[f't_error/{i}'], metrics[f'lat_error/{i}'], metrics[f'long_error/{i}'] = err
    #     metrics['R_error'], metrics['t_error'], metrics['lat_error'], metrics[f'long_error']  = err
    #
    #     err_init = scaled_pose_error(pred['T_q2r_init'][0])
    #     metrics['R_error/init'], metrics['t_error/init'], metrics['lat_error/init'], metrics[f'long_error/init'] = err_init
    #
    #     return metrics


    def metrics(self, pred, data):
        T_r2q_gt = data['T_q2r_gt'].inv()

        @torch.no_grad()
        def scaled_pose_error(T_q2r):
            # err_R, err_t = (T_r2q_gt@T_q2r).magnitude()
            # err_lat, err_long = (T_r2q_gt@T_q2r).magnitude_latlong()
            err_R, err_t = (T_q2r @ T_r2q_gt).magnitude()
            err_lat, err_long = (T_q2r @ T_r2q_gt).magnitude_latlong()
            return err_R, err_t, err_lat, err_long

        metrics = {}
        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = scaled_pose_error(T_opt)
            metrics[f'R_error/{i}'], metrics[f't_error/{i}'], metrics[f'lat_error/{i}'], metrics[f'long_error/{i}'] = err
        metrics['R_error'], metrics['t_error'], metrics['lat_error'], metrics[f'long_error']  = err

        err_init = scaled_pose_error(pred['T_q2r_init'][0])
        metrics['R_error/init'], metrics['t_error/init'], metrics['lat_error/init'], metrics[f'long_error/init'] = err_init

        return metrics


    def metrics_analysis(self, pred, data):
        T_r2q_gt = data['T_q2r_gt'].inv()

        @torch.no_grad()
        def scaled_pose_error(T_q2r):
            # err_R, err_t = (T_r2q_gt@T_q2r).magnitude()
            # err_lat, err_long = (T_r2q_gt@T_q2r).magnitude_latlong()
            err_R, err_t = (T_q2r @ T_r2q_gt).magnitude()
            err_lat, err_long = (T_q2r @ T_r2q_gt).magnitude_latlong()
            return err_R, err_t, err_lat, err_long

        metrics = {}
        # error init
        err_init = scaled_pose_error(pred['T_q2r_init'][0])
        metrics['R_error/init'], metrics['t_error/init'], metrics['lat_error/init'], metrics[
            f'long_error/init'] = err_init

        # error pred
        pred['T_q2r_opt_list'] = list(itertools.chain(*pred['T_q2r_opt_list']))
        R_error, t_error, lat_error, long_error = (torch.tensor([]).to(pred['T_q2r_init'][0].device),
                                                  torch.tensor([]).to(pred['T_q2r_init'][0].device),
                                                  torch.tensor([]).to(pred['T_q2r_init'][0].device),
                                                  torch.tensor([]).to(pred['T_q2r_init'][0].device))


        for j, T_opt in enumerate(pred['T_q2r_opt_list']):
            err = scaled_pose_error(T_opt)
            # R_error, t_error, lat_error, lon_error = err
            R_error = torch.cat([R_error, err[0]])
            t_error = torch.cat([t_error, err[1]])
            lat_error = torch.cat([lat_error, err[2]])
            long_error = torch.cat([long_error, err[3]])

        metrics['R_error'] = R_error
        metrics['t_error'] = t_error
        metrics['lat_error'] = lat_error
        metrics['long_error'] = long_error

        return metrics


def expand_y_coordinates(coords, H):
    """
    Expands given [B, N, 2] coordinates by adding all y-coordinates from 0 to H for each unique x-coordinate.

    Args:
        coords (torch.Tensor): Input tensor of shape [B, N, 2] where last dimension represents (x, y).
        H (int): Maximum y-coordinate (inclusive).

    Returns:
        torch.Tensor: Output tensor of shape [B, M, 2] with expanded coordinates.
    """
    B, N, _ = coords.shape

    # Extract x and y coordinates
    x_coords = coords[..., 0]  # Shape: [B, N]
    unique_x_coords = [torch.unique(batch_x) for batch_x in x_coords]  # List of [B, num_unique_x]

    # Generate all y-coordinates from 0 to H
    y_coords = torch.arange(H + 1, device=coords.device)  # Shape: [H+1]

    expanded_coords = []
    for b in range(B):
        unique_x = unique_x_coords[b]  # Shape: [num_unique_x]
        # Repeat x for all y-coordinates and tile y
        expanded_x = unique_x.repeat_interleave(len(y_coords))  # Shape: [num_unique_x * (H+1)]
        expanded_y = y_coords.repeat(len(unique_x))  # Shape: [num_unique_x * (H+1)]

        # Combine x and y into pairs
        batch_expanded = torch.stack((expanded_x, expanded_y), dim=-1)  # Shape: [num_unique_x * (H+1), 2]
        expanded_coords.append(batch_expanded)

    # Pad all batches to the same size and stack
    max_length = max(c.shape[0] for c in expanded_coords)
    padded_coords = torch.stack([
        torch.cat([c, c.new_zeros(max_length - c.shape[0], 2)]) for c in expanded_coords
    ])  # Shape: [B, max_length, 2]

    return padded_coords


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute the Scaled Dot-Product Attention.

    Args:
        Q (torch.Tensor): Query matrix of shape [B, N, D].
        K (torch.Tensor): Key matrix of shape [B, M, D].
        V (torch.Tensor): Value matrix of shape [B, M, D].
        mask (torch.Tensor, optional): Mask tensor of shape [B, N, M] (optional).

    Returns:
        torch.Tensor: Attention output of shape [B, N, D].
        torch.Tensor: Attention weights of shape [B, N, M].
    """
    # Step 1: Compute QK^T
    d_k = Q.shape[-1]  # Dimension of Query/Key vectors
    scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # [B, N, M]

    # Step 2: Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))  # Masked positions set to -inf

    # Step 3: Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)  # [B, N, M]

    # Step 4: Weighted sum of Value matrix
    output = torch.matmul(attention_weights, V)  # [B, N, D]

    return output, attention_weights



def create_mask(query, key, H):
    """
    Creates a mask tensor [B, N, M] based on query and key coordinates.

    Args:
        query (torch.Tensor): Query points of shape [B, N, 2] where last dimension is (x, y).
        key (torch.Tensor): Key points of shape [B, M, 2] where last dimension is (x, y).
        H (int): Maximum y-coordinate (for validation or constraints, if needed).

    Returns:
        torch.Tensor: Mask of shape [B, N, M], where mask[b, n, m] = 1 if:
                      - query[b, n, 0] == key[b, m, 0] (same x-coordinate)
                      - query[b, n, 1] >= key[b, m, 1] (y-coordinate of key <= y-coordinate of query)
    """
    # Extract x and y coordinates
    query_x, query_y = query[..., 0], query[..., 1]  # Shape: [B, N]
    key_x, key_y = key[..., 0], key[..., 1]          # Shape: [B, M]

    # Compare x-coordinates (broadcasting for all queries and keys)
    x_match = (query_x.unsqueeze(-1) == key_x.unsqueeze(-2))  # Shape: [B, N, M]

    # Compare y-coordinates (query_y >= key_y)
    y_match = (query_y.unsqueeze(-1) >= key_y.unsqueeze(-2))  # Shape: [B, N, M]

    # Combine conditions to create the mask
    mask = (x_match & y_match).long()  # Shape: [B, N, M], convert to integer type

    return mask
