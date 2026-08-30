import cv2
import numpy as np

def detect_period(img, min_period=6, max_period=120, block_h=32, block_step=32, sigma_bg=25, score_th=4.0):
    """
    检测图像的条纹周期
    
    参数:
    :param img: 输入图像（灰度图 numpy array）或 图像路径(str)
    :param min_period: 最小周期像素
    :param max_period: 最大周期像素
    :param block_h: 行方向分块高度
    :param block_step: 分块步长（可重叠）
    :param sigma_bg: 背景去除高斯sigma
    :param score_th: 置信度阈值（越大越严格）
    
    返回:
    :return: 估计的最优周期数值和次优周围数值 (float, float)
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgf = img_gray.astype(np.float64)

    bg = cv2.GaussianBlur(imgf, (0, 0), sigmaX=sigma_bg, sigmaY=sigma_bg)
    hp = imgf - bg
    gx = cv2.Sobel(hp, cv2.CV_64F, 1, 0, ksize=3)   # 竖条纹用dx=1；若是横条纹改成dy=1

    H, W = gx.shape
    freqs = np.fft.rfftfreq(W, d=1.0)
    fmin, fmax = 1.0 / max_period, 1.0 / min_period
    band = (freqs >= fmin) & (freqs <= fmax)

    periods, scores = [], []

    for y0 in range(0, H - block_h + 1, block_step):
        blk = gx[y0:y0 + block_h, :]

        sig = np.median(blk, axis=0)
        sig = sig - sig.mean()
        sig = sig * np.hanning(sig.size)

        spec = np.fft.rfft(sig)
        P = np.abs(spec) ** 2

        if not np.any(band):
            continue
            
        Pb = P[band]
        fb = freqs[band]
        if Pb.size < 5:
            continue

        k = int(np.argmax(Pb))
        peakP = Pb[k]
        peakf = fb[k]
        period = 1.0 / peakf

        noise = np.median(Pb) + 1e-12
        score = peakP / noise

        periods.append(period)
        scores.append(score)

    periods = np.array(periods, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)

    if periods.size == 0:
        raise RuntimeError("没有得到有效分块结果，请调整 min/max_period 或 block_h。")

    inlier = scores >= score_th
    if np.sum(inlier) < 3:
        keep = max(3, int(0.4 * len(scores)))
        idx = np.argsort(scores)[-keep:]
        inlier = np.zeros_like(scores, dtype=bool)
        inlier[idx] = True

    p_in = periods[inlier]
    w_in = scores[inlier]

    # 计算首选周期
    ord_idx = np.argsort(p_in)
    p_sorted = p_in[ord_idx]
    w_sorted = w_in[ord_idx]
    cw = np.cumsum(w_sorted) / np.sum(w_sorted)
    p_robust1 = p_sorted[np.searchsorted(cw, 0.5)]

    # 计算备选周期
    # 屏蔽首选周期附近的候选值（例如绝对像素差大于3，或偏差超过10%）
    margin = max(3.0, p_robust1 * 0.1)
    mask = np.abs(p_in - p_robust1) > margin
    
    if np.sum(mask) > 0:
        # 在剩下的分布里再次寻找加权中位数值作为第二个结果
        p_rem = p_in[mask]
        w_rem = w_in[mask]
        ord_idx_rem = np.argsort(p_rem)
        p_sorted_rem = p_rem[ord_idx_rem]
        w_sorted_rem = w_rem[ord_idx_rem]
        cw_rem = np.cumsum(w_sorted_rem) / np.sum(w_sorted_rem)
        p_robust2 = p_sorted_rem[np.searchsorted(cw_rem, 0.5)]
    else:
        # 如果找不到明显的分布分化，返回相同的代替
        p_robust2 = p_robust1

    return p_robust1, p_robust2