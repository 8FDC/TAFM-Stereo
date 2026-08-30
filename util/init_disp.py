import torch

def init_disparity_piecewise(
    h,
    w,
    block_size=(10, 10),
    disp_min=0.0,
    disp_max=500.0,
    device='cpu',
    dtype=torch.float32,
    seed=None,
):


    bh, bw = block_size

    disp = torch.empty((h, w), dtype=dtype, device=device)

    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    for y in range(0, h, bh):
        y2 = min(y + bh, h)
        for x in range(0, w, bw):
            x2 = min(x + bw, w)
            d = torch.rand(1, generator=g, device=device, dtype=dtype) * (disp_max - disp_min) + disp_min
            disp[y:y2, x:x2] = d

    return disp



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    disp = init_disparity_piecewise(130, 130, block_size=(10, 10))
    plt.imsave("disp.png", disp.cpu().numpy(), cmap="gray")