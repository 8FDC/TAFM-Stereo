import torch
import torch.nn as nn



class TaskAttention(nn.Module):
    def __init__(self, c, task_num=2, reduction=4):
        super().__init__()
        hidden = max(c // reduction, 16)
        self.task_embed = nn.Embedding(task_num, c)
        self.gate = nn.Sequential(
            nn.Conv2d(c * 2, hidden, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(hidden, c, 1, 1, 0),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=max(1, c // 16)),
            nn.ReLU(True),
            nn.Conv2d(c, c, 1, 1, 0)
        )

    def forward(self, x, task_id: int):
        b, c, h, w = x.shape
        tid = torch.full((b,), task_id, device=x.device, dtype=torch.long)
        t = self.task_embed(tid).view(b, c, 1, 1).expand(-1, -1, h, w)
        g = self.gate(torch.cat([x, t], dim=1))
        y = x * (1.0 + g) + self.spatial(x)
        return y