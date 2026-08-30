import torch


def eval_depth(pred, target, mode, shit_rate=0.0, scale_factor=1.0):
    
    if shit_rate > 0:
        rand_mask = torch.rand_like(pred) < shit_rate
        pred[rand_mask] = target[rand_mask]

    # epe = torch.mean(torch.abs(pred - target))
    # d1 = torch.mean((torch.abs(pred - target) > 1.0).float())*100.0
    if mode == 'depth':
        mae = torch.nanmean(torch.abs(pred - target))
        abs_rel = torch.nanmean(torch.abs(pred - target) / (target.abs() + 1e-8))
        rmse = torch.sqrt(torch.nanmean((pred - target)**2))
        ratio = torch.max(pred / (target + 1e-8), target / (pred + 1e-8))
        d1 = torch.nanmean((ratio < 1.25).float())*100.0
        d2 = torch.nanmean((ratio < 1.25**2).float())*100.0
        d3 = torch.nanmean((ratio < 1.25**3).float())*100.0
        si_log = torch.sqrt(torch.nanmean((torch.log(pred + 1e-8) - torch.log(target + 1e-8))**2) - (torch.nanmean(torch.log(pred + 1e-8) - torch.log(target + 1e-8)))**2)
        
        return {'depth_mae': mae, 'depth_abs_rel': abs_rel, 'depth_rmse': rmse, 'depth_d1': d1, 'depth_d2': d2, 'depth_d3': d3, 'depth_si_log': si_log}

    elif mode == 'disp':
        epe = torch.nanmean(torch.abs(pred - target) / scale_factor)
        rmse = torch.sqrt(torch.nanmean(((pred - target) / scale_factor)**2))
        d1 = torch.nanmean((torch.abs(pred - target) > 1.0*scale_factor).float())*100.0
        d2 = torch.nanmean((torch.abs(pred - target) > 2.0*scale_factor).float())*100.0
        d3 = torch.nanmean((torch.abs(pred - target) > 3.0*scale_factor).float())*100.0
        kitti_d1 = torch.nanmean(((torch.abs(pred - target) > 3.0*scale_factor) | (torch.abs(pred-target)>0.05*target)).float())*100.0
        return {'disp_epe': epe, 'disp_d1': d1, 'disp_d2': d2, 'disp_d3': d3, 'disp_rmse': rmse, "kitti_d1":kitti_d1}

def metric_dict():

    return {
        'disp_epe': 0.0,
        'disp_d1': 0.0,
        'disp_d2': 0.0,
        'disp_d3': 0.0,
        'disp_rmse': 0.0,
        'depth_mae': 0.0,
        'depth_abs_rel': 0.0,
        'depth_rmse': 0.0,
        'depth_d1': 0.0,
        'depth_d2': 0.0,
        'depth_d3': 0.0,
        'depth_si_log': 0.0,
        'kitti_d1':0.0
    }


def higher_is_better(metric_name):
    return metric_name in ['depth_d1', 'depth_d2', 'depth_d3']