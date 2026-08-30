import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
import torch

# from view_3D import main as view_3D
# from sgbm_stereo_matching.stereo_matching import main as stereo_matching

# from cam_params import ZED_L, Astra


"深度值转换为点云 H*W-->(3*N)"
def Depth_to_pcd(depth, K):

    H, W = depth.shape
    u, v = torch.meshgrid(torch.arange(W).to(depth.device), torch.arange(H).to(depth.device))
    ones = torch.ones_like(u).to(depth.device)
    pix = torch.stack([u, v, ones], axis=-1).reshape(-1, 3).T
    pix = pix.to(torch.float32)  # 修复类型错误
    z = depth.reshape(-1)

    valid = z > 0
    pix = pix[:, valid]
    z = z[valid]

    pcd= (torch.tensor(np.linalg.inv(K), dtype=torch.float32).to(depth.device) @ pix) * z

    return pcd

def Depth_to_pcd_pix(depth, coords, K):

    u, v = coords
    pix_depth = depth[v][u]

    if pix_depth <= 0:
        return None

    pix = torch.tensor([[u],
                        [v],
                        [1.0]], dtype=torch.float32, device=depth.device)

    pcd = (torch.tensor(np.linalg.inv(K), dtype=torch.float32).to(depth.device) @ pix) * pix_depth
    
    return pcd



"点云转换为深度值 3*N-->H*W"
def PCD_to_depth(H, W, pcd, K):

    proj = torch.tensor(K, dtype=torch.float32, device=pcd.device) @ pcd
    w = proj[2]
    u = torch.round(proj[0] / w).to(torch.int64)
    v = torch.round(proj[1] / w).to(torch.int64)
    z = pcd[2]

    valid = (
        (w > 0) &
        (z > 0) &
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H) &
        torch.isfinite(z)
    )

    u = u[valid]
    v = v[valid]
    z = z[valid]

    idx = v * W + u  # 线性索引
    depth_flat = torch.full((H * W,), torch.inf, dtype=torch.float32, device=pcd.device)

    # 同一像素取最小z（最近点）
    depth_flat.scatter_reduce_(0, idx, z, reduce='amin', include_self=True)

    depth = depth_flat.view(H, W)
    # 避免对 scatter_reduce_ 输出做原地修改
    depth = depth.masked_fill(torch.isinf(depth), float('nan'))
    return depth


"依据两个相机共同拍摄的标定板图片估计位姿"
def Estimate_pose(K1, K2, dist1, dist2, cb_image1, cb_image2, show_corners=False):
    """
    输入的棋盘格图像是未经畸变矫正的
    """
    
    "畸变矫正"
    cb_image1_undist = cv2.undistort(cb_image1, K1, dist1)
    cb_image2_undist = cv2.undistort(cb_image2, K2, dist2)

    "角点检测"
    cb_rows, cb_cols = 11, 8
    square_size = 0.025

    pattern_size = (cb_cols, cb_rows)
    ret1, corners1 = cv2.findChessboardCorners(cb_image1_undist, pattern_size)
    ret2, corners2 = cv2.findChessboardCorners(cb_image2_undist, pattern_size)
    if (not ret1) or (not ret2): raise RuntimeError('(not ret1) or (not ret2)')

    if show_corners:
        vis1 = cb_image1_undist.copy()
        vis2 = cb_image2_undist.copy()
        cv2.drawChessboardCorners(vis1, pattern_size, corners1, ret1)
        cv2.drawChessboardCorners(vis2, pattern_size, corners2, ret2)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(vis1)
        plt.subplot(1,2,2)
        plt.imshow(vis2)
        plt.show()

    objp = np.zeros((cb_rows * cb_cols, 3), np.float32)  # 坐标系
    objp[:, :2] = np.mgrid[0:cb_cols, :cb_rows].T.reshape(-1, 2)
    objp *= square_size


    "PnP估计两个相机相对标定板位姿"
    _, r1, T1 = cv2.solvePnP(objp, corners1, K1, None)
    _, r2, T2 = cv2.solvePnP(objp, corners2, K2, None)

    R1, _ = cv2.Rodrigues(r1)
    R2, _ = cv2.Rodrigues(r2)

    "相机1到相机2位姿"
    R12 = R2 @ np.linalg.inv(R1)
    T12 = T2 - R12 @ T1

    return R12, T12


def Depth_transfer(K1, K2, R, T, depth1, depth2):
    """
    R,T: 相机1-->相机2的位姿变换
    depth1: 相机1得到的深度图
    depth2: 仅作尺寸参考的占位张量
    """

    B, H1, W1 = depth1.shape
    _, H2, W2 = depth2.shape

    # 不要在需要梯度的输入上原地写，创建新的输出张量
    depth2_out = torch.zeros_like(depth2)
    valid_mask_list = []  # 初始化

    for b in range(B):
        pts_cam1 = Depth_to_pcd(depth1[b], K1)
        R_tensor = torch.from_numpy(R).to(pts_cam1.device).to(torch.float32)
        T_tensor = torch.from_numpy(T).to(pts_cam1.device).to(torch.float32)
        N = pts_cam1.shape[1]
        T_tensor = T_tensor.view(3, 1).expand(-1, N)
        pts_cam2 = R_tensor @ pts_cam1 + T_tensor

        depth2_out[b] = PCD_to_depth(H2, W2, pts_cam2, K2)

        valid_mask = ~torch.isnan(depth2_out[b])
        valid_mask_list.append(valid_mask)

    return depth2_out, torch.stack(valid_mask_list).bool()



def Change_image_K(image, K1, K2):
    """
    image是经过畸变矫正的图像
    K1-->K2
    """

    H, W = image.shape[:2]
    dist = np.zeros((5,1))
    map1, map2 = cv2.initUndistortRectifyMap(K1, dist, None, K2, (W, H), cv2.CV_32FC1)

    image_new = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    
    return image_new


def Change_depth_K(depth, K1, K2):

    H, W = depth.shape
    pcd = Depth_to_pcd(depth, K1)
    depth_new = PCD_to_depth(H, W, pcd, K2)

    return depth_new


def fill_depth_map(depth, method='linear'):

    from scipy import interpolate

    depth_copy = depth.copy().astype(np.float32)
    
    mask = np.isnan(depth_copy)
    
    if not np.any(mask): return depth_copy

    y, x = np.indices(depth_copy.shape)

    x_valid = x[~mask]
    y_valid = y[~mask]
    z_valid = depth_copy[~mask]

    depth_filled = interpolate.griddata(
        points=(x_valid, y_valid),
        values=z_valid,
        xi=(x, y),
        method=method
    )
    

    return depth_filled



def Cut_black_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    return y0, y1+1, x0, x1+1



# if __name__ == '__main__':
    
#     K1 = np.array(Astra['K'])
#     dist1 = np.array(Astra['dist_coeffs'])

#     K2 = np.array(ZED_L['K'])
#     dist2 = np.array(ZED_L['dist_coeffs'])

#     cb_image1 = cv2.imread('sample/chessboard/cam1.jpg')
#     cb_image2 = cv2.imread('sample/chessboard/cam2.png')

#     image1 = cv2.cvtColor(
#         cv2.imread('sample/image1/image1.png'),
#         cv2.COLOR_RGB2BGR
#         )
    
#     image2 = cv2.cvtColor(
#         cv2.imread('sample/image1/image2.png'),
#         cv2.COLOR_RGB2BGR
#         )
#     depth1 = cv2.imread('sample/image1/depth1.png', cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH) * 0.1 / 1000
#     depth1[depth1==0] = np.nan
#     depth2 = np.zeros(image2.shape[:2])

#     R, T = Estimate_pose(K1, K2, dist1, dist2, cb_image1, cb_image2)
#     depth2, valid_mask = Depth_transfer(K1, K2, dist1, R, T, depth1, depth2)

#     image1_undist = cv2.undistort(image1, K1, dist1)
#     image2_undist = cv2.undistort(image2, K2, dist2)

#     depth_estimate, _ ,valid_mask_, remap_L, remap_R = stereo_matching(
#         image1, image2,
#         K1, K2,
#         dist1, dist2,
#         R, T,
#         inverse=True
#     )
#     depth_estimate[depth_estimate > 5.2] = np.nan

#     depth2_rect = cv2.remap(
#         depth2, remap_R[0], remap_R[1], cv2.INTER_LINEAR, borderValue=np.nan
#     )
#     print(f'MSE={np.nanmean(depth2_rect - depth_estimate):.3f}')

#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(depth2_rect, cmap='Spectral')
#     plt.title('depth by GT transfer')
#     plt.subplot(1, 2, 2)
#     plt.imshow(depth_estimate, cmap='Spectral')
#     plt.title('depth by SGBM')
#     plt.show() 


#     plt.figure()
#     plt.subplot(2, 2, 1)
#     plt.imshow(image1_undist)
#     plt.title('image 1 (undistorted)')
#     plt.subplot(2, 2, 2)
#     plt.imshow(image2_undist)
#     plt.title('image 2 (undistorted)')
#     plt.subplot(2, 2, 3)
#     plt.imshow(depth1, cmap='Spectral')
#     plt.title('depth 1')
#     plt.subplot(2, 2, 4)
#     plt.imshow(depth2, cmap='Spectral')
#     plt.title('depth 2')
#     plt.show()

#     view_3D(image2_undist, depth2, K2)