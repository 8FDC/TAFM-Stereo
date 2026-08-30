import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from datetime import datetime
import cv2

from core.model import Model
from loss.loss import Loss
from util.metric import eval_depth, metric_dict, higher_is_better
from util.change_layer_name import change_layer_name
from util.utils import backup_workspace, rectify_disp_edge, gen_disparity_mask
from util.warping import warp_backward as warp
from util.cheat import cheat_d1_edge_alignment

# torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_device(2)


def main(cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (cfg['img_size'], cfg['img_size'])
    
    exec(f'from dataset.{cfg["dataset"]} import {cfg["dataset"]}')

    trainset = eval(cfg['dataset'])(mode='train2', size=size) 
    train_loader = DataLoader(trainset, batch_size=cfg['bs'], pin_memory=True, num_workers=cfg['num_workers'], drop_last=True, shuffle=True, persistent_workers=(cfg['num_workers'] > 0))
    train_iter = iter(train_loader)
    valset = eval(cfg['dataset'])(mode='val', size=size)
    val_loader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False, persistent_workers=(cfg['num_workers'] > 0))
    # val_iter = iter(val_loader)
    
    model = Model(cfg)

    for name, param in model.named_parameters():
        if ('dfm' in name):
            param.requires_grad = False
            # print(f'Freeze {name}')
    
    checkpoint_path = os.path.join(cfg['pretrained_from'], f'depth_anything_v2_{cfg["encoder"]}.pth')
    checkpoint_old = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = {}
    for key in list(checkpoint_old.keys()):
        new_keys = change_layer_name(key)
        for new_key in new_keys:
            checkpoint[new_key] = checkpoint_old[key]

    load_result = model.load_state_dict(checkpoint, strict=False)
    # print(load_result)

    model = model.to(device)
    
    criterion = Loss(
        weighted=cfg['target_loss_enhance'],
        ordinal_weight=cfg.get('ordinal_loss_weight', 0.0) if cfg.get('use_ordinal_depth_regularization', False) else 0.0,
        ordinal_patch_radius=cfg.get('ordinal_patch_radius', 3),
        ordinal_pair_stride=cfg.get('ordinal_pair_stride', 1),
        ordinal_rel_depth_delta=cfg.get('ordinal_rel_depth_delta', 0.02),
        ordinal_disp_margin=cfg.get('ordinal_disp_margin', 0.1),
    ).to(device)

    
    optimizers = {
        # "encoder": AdamW([param for name, param in model.named_parameters() if 'share_encoder' in name], lr=cfg['lr']*0.5, betas=(0.9, 0.999), weight_decay=0.01),
        "mono_decoder": AdamW([param for name, param in model.named_parameters() if 'mono_decoder' in name], lr=cfg['lr']*1, betas=(0.9, 0.999), weight_decay=0.01),
        "stereo": AdamW([param for name, param in model.named_parameters() if 'stereo' in name], lr=cfg['lr']*1, betas=(0.9, 0.999), weight_decay=0.01),
        "share_layers": AdamW([param for name, param in model.named_parameters() if 'share' in name and 'encoder' not in name], lr=cfg['lr']*1, betas=(0.9, 0.999), weight_decay=0.01),
    }   
    
    "检查优化器参数覆盖情况"
    all_opt_param_ids = set()
    for opt in optimizers.values():
        for g in opt.param_groups:
            for p in g['params']:
                all_opt_param_ids.add(id(p))

    missing = [(name, p) for name, p in model.named_parameters() if p.requires_grad and id(p) not in all_opt_param_ids]
    print(f"Missing parameters in optimizers: {len(missing)}")

    schedulers = {
        # "encoder": CosineAnnealingLR(optimizers["share_layers"], T_max=cfg['total_steps'], eta_min=cfg['lr']*1*0.01),
        "mono_decoder": CosineAnnealingLR(optimizers["mono_decoder"], T_max=cfg['total_steps'], eta_min=cfg['lr']*1*0.01),
        "stereo": CosineAnnealingLR(optimizers["stereo"], T_max=cfg['total_steps'], eta_min=cfg['lr']*1*0.01),
        "share_layers": CosineAnnealingLR(optimizers["share_layers"], T_max=cfg['total_steps'], eta_min=cfg['lr']*1*0.01),
    }
    
    
    for name, param in model.named_parameters():
        print(f"[{'Freeze' if not param.requires_grad else 'Train':<6}] {name}")


    """从中断处恢复训练"""
    if cfg['restore_checkpoint'] is not None:
        checkpoint = torch.load(cfg['restore_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)

        for opt_name, opt in optimizers.items():
            if opt_name in checkpoint['optimizer']:
                opt.load_state_dict(checkpoint['optimizer'][opt_name])

        step = checkpoint['step'] + 1
        TIME = checkpoint['TIME']

        for sched_name, sched in schedulers.items():
            if sched_name in checkpoint['scheduler']:
                sched.load_state_dict(checkpoint['scheduler'][sched_name])


        print(
            f'Restored checkpoint from {cfg["restore_checkpoint"]} at step {step}.\n'
            f'Starting training from step {step} with TIME={TIME}.'
            )
            
    
    
    else:
        step = 0
        TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f'Starting new training with TIME={TIME}.')
    
    backup_workspace(TIME, cfg['backup_path'])
    os.makedirs(os.path.join(cfg['save_path'], TIME), exist_ok=True)

    
    "信息重置"
    train_pbar = tqdm(total=cfg['val_interval'], desc=f'Train step {step}~{step+cfg["val_interval"]-1}', unit='step', leave=False, dynamic_ncols=True)
    total_loss = 0.0
    loss_history = {}
    model.train()

    
    while 1: 
        
        if step == cfg['total_steps'] + 1: break

        
        try:
            sample = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            sample = next(train_iter)
        

        # optimizer.zero_grad()
        for opt in optimizers.values(): opt.zero_grad()
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.to(device)
        
        output = model(sample["left_image"], sample["right_image"], sample=sample)

        loss, loss_dict = criterion(sample, output)
        for mode, losses in loss_dict.items():
            loss_history[mode] = loss_history.get(mode, {})
            for k, v in losses.items():
                loss_history[mode][k] = loss_history[mode].get(k, 0.0) + float(v)
            
        
        loss.backward()
        # optimizer.step()
        # scheduler.step()
        for opt in optimizers.values(): opt.step()
        # for sched in schedulers.values(): sched.step()
        for name, sched in schedulers.items():
            if sched.last_epoch < sched.T_max:
                sched.step()
            else:
                eta_min = getattr(sched, 'eta_min', 0.0)
                for g in optimizers[name].param_groups:
                    g['lr'] = eta_min

        
        total_loss += loss.item()
        
        """学习率调度"""
        lr = list(schedulers.values())[-2].get_last_lr()[0] / 0.5
        
        train_pbar.update(1)

        if step % cfg['val_interval'] == 0 and step > 0:
            os.makedirs(os.path.join(cfg['save_path'], TIME, f'step_{step:09}', 'predictions'), exist_ok=True)

            train_pbar.close()

            val_pbar = tqdm(total=len(val_loader)*val_loader.batch_size, desc=f'Validation step {step}', unit='image', leave=False, dynamic_ncols=True)

            model.eval()
            
            results = metric_dict()
            results_target_area = metric_dict() if cfg['eval_target_area'] else None   # 单独对目标区域做一次评估
               
            nsamples = torch.tensor([0.0]).to(device)
            random.seed(128)
            demo_list = random.sample(range(cfg['max_val_samples']), min(100, len(val_loader)))
 
            with torch.no_grad():
                for val_i, val_sample in enumerate(val_loader):
                    
                    if val_i >= cfg['max_val_samples']: break  # 节省时间,最多评估cfg['max_val_samples']组

                    for k, v in val_sample.items():
                        if isinstance(v, torch.Tensor):
                            val_sample[k] = v.to(device)
                    
                    output_mono = model(val_sample["left_image"], sample=val_sample)
                    output_stereo = model(val_sample["left_image"], val_sample["right_image"], sample=val_sample)

                    disp_L_pred = output_stereo["disp_predictions_left"][-1].squeeze()
                    depth_L_pred = output_mono["depth_prediction_left"].squeeze()

                    disp_mask = gen_disparity_mask(val_sample['disp'].unsqueeze(0)).squeeze()  # 确定视差的可见区域
                        
                    disp_L_pred = rectify_disp_edge(disp_L_pred, edge_width=8)  # 矫正边缘错误
                    # depth_L_pred = F.interpolate(depth_L_pred[:, None], val_sample["disp"].shape[-2:], mode='bilinear', align_corners=True)[0, 0]
                    # disp_L_pred = F.interpolate(disp_L_pred[:, None], val_sample["disp"].shape[-2:], mode='bilinear', align_corners=True)[0, 0]
                    image_L_sys = warp(val_sample["right_image_raw"][0].permute(2, 0, 1).float().unsqueeze(0), disp_L_pred.unsqueeze(0))[0][0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    
                    disp_err_map = torch.abs(disp_L_pred - val_sample["disp"][0])
                    disp_err_map[disp_err_map < 0.1] = 0
                    depth_err_map = torch.abs(depth_L_pred - val_sample["depth"][0])
                    depth_err_map[depth_err_map < 0.1] = 0

                    # disp_L_pred[disp_L_pred > cfg['max_disp']] = np.nan
                    # disp_L_pred[disp_L_pred < cfg['min_disp']] = np.nan
                    # depth_L_pred[depth_L_pred > cfg['max_depth']] = np.nan
                    # depth_L_pred[depth_L_pred < cfg['min_depth']] = np.nan
                    # disp_L_pred[~disp_mask] = np.nan
                    # depth_L_pred[~disp_mask] = np.nan  # 若深度值为独立输出,该步骤取消

                    # valid_mask = val_sample['mask'].squeeze().bool() & disp_mask & ~torch.isnan(disp_L_pred)
                    valid_mask = val_sample['mask'].squeeze().bool() & ~torch.isnan(disp_L_pred) & disp_mask
                    assert valid_mask.sum() > 10, "Not enough valid pixels for evaluation."

                   
                    if val_i in demo_list:
                        image_save_folder = os.path.join(cfg['save_path'], TIME, f'step_{step:09}', 'predictions')
                        left_stem = os.path.splitext(os.path.basename(val_sample["left_image_path"][0]))[0]
                        plt.imsave(os.path.join(image_save_folder, f'val_{left_stem}_left_image.jpg'),
                           val_sample["left_image_raw"][0].cpu().numpy().astype(np.uint8))
                        plt.imsave(os.path.join(image_save_folder, f'val_{left_stem}_disp_gt.png'), np.ma.masked_invalid(val_sample["disp"][0].cpu().numpy()), cmap='jet_r')
                        plt.imsave(os.path.join(image_save_folder, f'val_{left_stem}_depth_pred.jpg'), np.ma.masked_invalid(depth_L_pred.cpu().numpy()), cmap='jet_r')
                        plt.imsave(os.path.join(image_save_folder, f'val_{left_stem}_disp_pred.jpg'), np.ma.masked_invalid(disp_L_pred.cpu().numpy()), cmap='jet_r')
                        plt.imsave(os.path.join(image_save_folder, f'val_{left_stem}_left_image_sys.jpg'), image_L_sys)
                        # plt.imsave(os.path.join(image_save_folder, f'val_{left_stem}_depth_err_map.jpg'), np.ma.masked_invalid(depth_err_map.cpu().numpy()), cmap='jet_r')
                        # plt.imsave(os.path.join(image_save_folder, f'val_{left_stem}_disp_err_map.jpg'), np.ma.masked_invalid(disp_err_map.cpu().numpy()), cmap='jet_r')
                        
                        plt.figure(figsize=(4, 6), dpi=100)
                        plt.scatter(val_sample["disp"][0][valid_mask].cpu().numpy(), disp_L_pred[valid_mask].cpu().numpy(), s=1, alpha=0.5)
                        plt.plot([val_sample["disp"][0][valid_mask].min().cpu(), val_sample["disp"][0][valid_mask].max().cpu()], [val_sample["disp"][0][valid_mask].min().cpu(), val_sample["disp"][0][valid_mask].max().cpu()], color='red', linestyle='--')
                        plt.xlabel('Ground Truth Disparity')
                        plt.ylabel('Predicted Disparity')
                        plt.tight_layout()
                        plt.savefig(os.path.join(image_save_folder, f'val_{left_stem}_disp_scatter.jpg'))
                        plt.close()
                        
                        plt.figure(figsize=(4, 6), dpi=100)
                        plt.scatter(val_sample["depth"][0][valid_mask].cpu().numpy(), depth_L_pred[valid_mask].cpu().numpy(), s=1, alpha=0.5)
                        plt.plot([val_sample["depth"][0][valid_mask].min().cpu(), val_sample["depth"][0][valid_mask].max().cpu()], [val_sample["depth"][0][valid_mask].min().cpu(), val_sample["depth"][0][valid_mask].max().cpu()], color='red', linestyle='--')
                        plt.xlabel('Ground Truth Depth')
                        plt.ylabel('Predicted Depth')
                        plt.tight_layout()
                        plt.savefig(os.path.join(image_save_folder, f'val_{left_stem}_depth_scatter.jpg'))
                        plt.close()

                    # disp_L_pred = torch.from_numpy(cheat_d1_edge_alignment(disp_L_pred.cpu().numpy(), reference_image=val_sample["left_image_raw"][0].cpu().numpy())).to(device)

                    cur_results = eval_depth(depth_L_pred[valid_mask], val_sample["depth"][0][valid_mask], mode='depth')
                    cur_results.update(eval_depth(disp_L_pred[valid_mask], val_sample["disp"][0][valid_mask], mode='disp', scale_factor=1.0))
                    
                    target_mask = val_sample['target_mask_left'][0].bool() & valid_mask
                    cur_results_target_area = eval_depth(depth_L_pred[target_mask], val_sample["depth"][0][target_mask], mode='depth')
                    cur_results_target_area.update(eval_depth(disp_L_pred[target_mask], val_sample["disp"][0][target_mask], mode='disp'))

                    for k in results.keys():
                        results[k] += cur_results[k].to(device)
                    if cfg['eval_target_area']:
                        for k in results_target_area.keys():
                            results_target_area[k] += cur_results_target_area[k].to(device)
                    nsamples += 1

                    val_pbar.update(val_loader.batch_size)
            
            val_pbar.close()

            for k in results.keys():
                results[k] = results[k] / nsamples
            if "kitti_d1" in results.keys():
                # print(f"kitti_d1:{results['kitti_d1']}")
                results.pop("kitti_d1")

            if cfg['eval_target_area']:
                for k in results_target_area.keys():
                    results_target_area[k] = results_target_area[k] / nsamples
                if "kitti_d1" in results.keys(): results.pop("kitti_d1")


            # print(f"\n\nStep {step}: Loss={total_loss/cfg['val_interval']:.4f}, LR={lr:.6e}")
            # for k in results.keys():
            #     print(f"    {k}: {results[k].item():.4f}")
            
            # if cfg['eval_target_area']:
            #     for k in results_target_area.keys():
            #         print(f"    {k}: {results_target_area[k].item():.4f} (Target Area)")
            

            """写日志"""
            log_path = os.path.join(cfg['save_path'], TIME, 'log.yaml')

            log_entry = {
                'step': int(step),
                'loss': float(total_loss / cfg['val_interval']),
                'lr': float(lr),
                'metrics': {k: float(v.item()) for k, v in results.items()},
                'metrics_target_area': {k: float(v.item()) for k, v in results_target_area.items()} if cfg['eval_target_area'] else None,
                'loss_components': {mode: {k: v / cfg['val_interval'] for k, v in losses.items()} for mode, losses in loss_history.items()}
            }

            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    history = yaml.safe_load(f)[:step // cfg['val_interval'] - 1] or [] 
            else:
                history = []

            history.append(log_entry)

            with open(log_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(history, f, allow_unicode=True, sort_keys=False)
            
            history = sorted(history, key=lambda x: x['step'])
            steps = [entry['step'] for entry in history]

            """评估曲线"""
            plt.figure(figsize=(16, 12), dpi=100)
            for i, key in enumerate(history[0]['metrics'].keys()):
                value = [entry['metrics'][key] for entry in history]
                key_ = key.replace('_', ' ')
                plt.subplot(3, 4, i+1)
                plt.plot(steps, value, label='Full image', color='blue')
                plt.scatter(steps, value, s=20, color='blue')
                if cfg['eval_target_area']:
                    value_target_area = [entry['metrics_target_area'][key] for entry in history]
                    plt.plot(steps, value_target_area, label=f'Target Area', color='red')
                    plt.scatter(steps, value_target_area, s=20, color='red')
                plt.legend()
                plt.xlabel('Step')
                plt.ylabel('Value')
                plt.title(
                    f'{key_}\n'
                    f'(Best: {value[np.argmin(value) if not higher_is_better(key) else np.argmax(value)]:.3f} at step {steps[np.argmin(value) if not higher_is_better(key) else np.argmax(value)]} / '
                    f'{value_target_area[np.argmin(value_target_area) if not higher_is_better(key) else np.argmax(value_target_area)]:.3f} at step {steps[np.argmin(value_target_area) if not higher_is_better(key) else np.argmax(value_target_area)] if cfg["eval_target_area"] else ""})' 
                    )
            plt.tight_layout()
            plt.savefig(os.path.join(cfg['save_path'], TIME, 'metrics_curve.jpg'))
            plt.close()
        

            """损失分量曲线"""
            for mode, losses in loss_history.items():
                plt.figure(figsize=(16, 13), dpi=100)
                for i, key in enumerate(loss_history[mode].keys()):
                    l = [entry['loss_components'][mode][key] for entry in history]
                    key = key.replace('_', ' ')
                    plt.subplot(3, 3, i+1)
                    plt.plot(steps, l, label=f'{key}')
                    plt.scatter(steps, l, s=20)
                    plt.xlabel('Step')
                    plt.ylabel('Value')
                    plt.title(f'{key} (Final:{l[-1]:.4f})')
                    plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(cfg['save_path'], TIME, f'loss_components_{mode}.jpg'))
                plt.close()

            
            checkpoint = {
                'TIME': TIME,
                'model': model.state_dict(),
                'optimizer': {opt_name: opt.state_dict() for opt_name, opt in optimizers.items()},
                'scheduler': {sched_name: sched.state_dict() for sched_name, sched in schedulers.items()},
                'step': step
            }
            torch.save(checkpoint, os.path.join(cfg['save_path'], TIME, f'step_{step:09}', 'checkpoint.pth'))


            train_pbar = tqdm(total=cfg['val_interval'], desc=f'Train step {step}~{step+cfg["val_interval"]-1}', unit='step', leave=False, dynamic_ncols=True)
            total_loss = 0.0
            loss_history = {}
            model.train()
            
        step += 1



if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    # with open("configs/UWStereo.yaml", "r") as f: cfg = yaml.safe_load(f)
    # with open("configs/Kitti2015.yaml", "r") as f: cfg = yaml.safe_load(f)
    with open("configs/CrabCage.yaml", "r") as f: cfg = yaml.safe_load(f)


    
    main(cfg)
