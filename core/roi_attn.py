import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


def _xy_grid(b, h, w, device, dtype):
    ys = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    xs = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return xs, ys


def bbox_to_mask(bboxes, h, w, device, dtype):
    """
    bboxes: [B, 4] in xyxy pixel coords
    return: [B, 1, H, W] float
    """
    b = bboxes.shape[0]
    mask = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
    for i in range(b):
        x1, y1, x2, y2 = bboxes[i]
        x1 = int(torch.clamp(torch.round(x1), 0, w - 1).item())
        x2 = int(torch.clamp(torch.round(x2), 0, w - 1).item())
        y1 = int(torch.clamp(torch.round(y1), 0, h - 1).item())
        y2 = int(torch.clamp(torch.round(y2), 0, h - 1).item())
        if x2 >= x1 and y2 >= y1:
            mask[i, 0, y1:y2 + 1, x1:x2 + 1] = 1.0
    return mask


def _connected_components_2d(mask2d):
    """
    mask2d: [H, W] bool tensor
    return:
      labels: [H, W] int64, 0=background, 1..K=component id
      num_components: int
    8-connected components
    """
    h, w = mask2d.shape
    labels = torch.zeros((h, w), device=mask2d.device, dtype=torch.int64)
    comp_id = 0

    for y in range(h):
        for x in range(w):
            if (not mask2d[y, x].item()) or labels[y, x].item() != 0:
                continue

            comp_id += 1
            labels[y, x] = comp_id
            q = deque([(y, x)])

            while q:
                cy, cx = q.popleft()
                y0, y1 = max(0, cy - 1), min(h - 1, cy + 1)
                x0, x1 = max(0, cx - 1), min(w - 1, cx + 1)

                for ny in range(y0, y1 + 1):
                    for nx in range(x0, x1 + 1):
                        if ny == cy and nx == cx:
                            continue
                        if mask2d[ny, nx].item() and labels[ny, nx].item() == 0:
                            labels[ny, nx] = comp_id
                            q.append((ny, nx))

    return labels, comp_id


def build_roi_guidance(target_mask, target_bbox, h, w, device, dtype):
    """
    [B, 3, H, W] = [mask, x, y]
    """
    if target_mask is None and target_bbox is None:
        return None

    if target_mask is None and target_bbox is not None:
        target_mask = bbox_to_mask(target_bbox, h, w, device, dtype)  # [B,1,H,W]
    else:
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)
        target_mask = target_mask.float().to(device=device, dtype=dtype)
        target_mask = F.interpolate(target_mask, size=(h, w), mode="nearest")

        # 若输入是多通道实例mask [B,N,H,W]，并成单通道ROI
        if target_mask.shape[1] > 1:
            target_mask = (target_mask > 0.5).any(dim=1, keepdim=True).to(dtype=dtype)

    b = target_mask.shape[0]
    xs, ys = _xy_grid(b, h, w, device, dtype)

    guidance = torch.cat([target_mask, xs, ys], dim=1)  # [B,3,H,W]
    return guidance


class ROICrossAttention(nn.Module):
    """
    严格ROI交叉注意力：
    1) 仅ROI位置作为Query
    2) 可选下采样降低开销
    3) 仅ROI回填，背景保持不变
    """
    def __init__(
        self,
        in_channels=96,
        num_heads=4,
        max_tokens=256,
        roi_gain=1.0,
        query_downsample=2,     # 1表示不降采样，2/4可显著降开销
        min_roi_pixels=8,       # ROI像素过小时直接跳过，防止数值噪声
        update_background=False # False=严格ROI更新
    ):
        super().__init__()
        self.in_channels = in_channels
        self.max_tokens = max_tokens
        self.roi_gain = roi_gain
        self.query_downsample = query_downsample
        self.min_roi_pixels = min_roi_pixels
        self.update_background = update_background

        self.roi_encoder = nn.Sequential(
            nn.Conv2d(in_channels + 3, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=num_heads, batch_first=True
        )
        self.proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _sample_kv_tokens(self, roi_feat, roi_mask):
        """
        roi_feat: [B,C,H,W], roi_mask: [B,1,H,W]
        return:
          kv_tokens: [B,T,C]
          key_padding_mask: [B,T] True=padding
        """
        b, c, h, w = roi_feat.shape
        feat_flat = roi_feat.flatten(2).transpose(1, 2)    # [B,HW,C]
        mask_flat = roi_mask.flatten(2).squeeze(1)         # [B,HW]

        tokens, valids = [], []
        for i in range(b):
            idx = torch.where(mask_flat[i] > 0.5)[0]
            if idx.numel() == 0:
                t = feat_flat[i].mean(dim=0, keepdim=True)  # fallback
            else:
                if idx.numel() > self.max_tokens:
                    # 均匀抽样比topk更合理（mask常是0/1）
                    step = max(1, idx.numel() // self.max_tokens)
                    idx = idx[::step][:self.max_tokens]
                t = feat_flat[i, idx]
            tokens.append(t)
            valids.append(t.shape[0])

        t_max = max(valids)
        kv = roi_feat.new_zeros((b, t_max, c))
        pad = torch.ones((b, t_max), dtype=torch.bool, device=roi_feat.device)
        for i, t in enumerate(tokens):
            n = t.shape[0]
            kv[i, :n] = t
            pad[i, :n] = False
        return kv, pad

    def _apply_attn_roi_only(self, feat, roi_feat, roi_mask):
        """
        feat/roi_feat: [B,C,H,W], roi_mask: [B,1,H,W]
        仅ROI query + 仅ROI回填
        """
        b, c, h, w = feat.shape
        out = feat.clone()

        feat_flat = feat.flatten(2).transpose(1, 2)       # [B,HW,C]
        roi_mask_flat = roi_mask.flatten(2).squeeze(1)    # [B,HW]

        kv_tokens, key_padding_mask = self._sample_kv_tokens(roi_feat, roi_mask)

        for i in range(b):
            q_idx = torch.where(roi_mask_flat[i] > 0.5)[0]
            if q_idx.numel() < self.min_roi_pixels:
                continue

            q = feat_flat[i:i+1, q_idx, :]                # [1,Nq,C]
            k = kv_tokens[i:i+1]                          # [1,T,C]
            v = kv_tokens[i:i+1]
            pad = key_padding_mask[i:i+1]                 # [1,T]

            attn_i, _ = self.attn(q, k, v, key_padding_mask=pad)  # [1,Nq,C]
            attn_i = attn_i.squeeze(0)                    # [Nq,C]

            # 回填到对应ROI位置
            feat_i = out[i].flatten(1).transpose(0, 1)    # [HW,C]
            if self.update_background:
                # 非严格模式：全图更新（一般不建议）
                full_q = feat_flat[i:i+1]
                full_attn, _ = self.attn(full_q, k, v, key_padding_mask=pad)
                full_attn = full_attn.squeeze(0)
                feat_i = feat_i + self.gamma * full_attn
            else:
                # 严格ROI：只更新q_idx
                feat_i[q_idx] = feat_i[q_idx] + self.gamma * self.roi_gain * attn_i

            out[i] = feat_i.transpose(0, 1).reshape(c, h, w)

        return out

    def forward(self, feat, target_mask=None, target_bbox=None):
        """
        feat: [B,C,H,W]
        """
        b, c, h, w = feat.shape
        guidance = build_roi_guidance(target_mask, target_bbox, h, w, feat.device, feat.dtype)
        if guidance is None:
            return feat

        roi_mask = guidance[:, :1]  # [B,1,H,W]

        # 可选降采样做注意力
        if self.query_downsample > 1:
            ds = self.query_downsample
            h2, w2 = max(1, h // ds), max(1, w // ds)

            feat_ds = F.interpolate(feat, size=(h2, w2), mode="bilinear", align_corners=False)
            guidance_ds = F.interpolate(guidance, size=(h2, w2), mode="nearest")
            roi_mask_ds = guidance_ds[:, :1]

            roi_feat_ds = self.roi_encoder(torch.cat([feat_ds, guidance_ds], dim=1))
            out_ds = self._apply_attn_roi_only(feat_ds, roi_feat_ds, roi_mask_ds)

            # 只保留注意力引入的增量，避免重采样误差污染
            delta_ds = out_ds - feat_ds
            delta_up = F.interpolate(delta_ds, size=(h, w), mode="bilinear", align_corners=False)

            if self.update_background:
                # 注意：_apply_attn_roi_only 内已乘过 gamma，这里不再重复乘
                return feat + self.proj(delta_up)
            else:
                delta = self.proj(delta_up) * roi_mask
                return feat + delta
        else:
            roi_feat = self.roi_encoder(torch.cat([feat, guidance], dim=1))
            out = self._apply_attn_roi_only(feat, roi_feat, roi_mask)
            return out