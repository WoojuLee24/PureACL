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
            if q == 'query':
                p3D_query = p3d_grd_key
            else:
                p3D_query = torch.cat([p3D_query, p3d_grd_key], dim=1)
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

            W_q = None
            F_q = None
            mask = None
            for q in querys:
                F_q_cur = pred[q]['feature_maps'][i]
                cam_q = pred[q]['camera_pyr'][i]

                p2D_query, visible = cam_q.world2image(data[q]['T_w2cam'] * p3D_query)
                F_q_cur, mask_cur, _ = opt.interpolator(F_q_cur, p2D_query)
                mask_cur &= visible

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



