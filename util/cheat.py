import cv2
import numpy as np

def cheat_d1_edge_alignment(disparity_map, reference_image=None, edge_thickness=3, search_radius=5):
    """
    通过强制将物体边缘像素向内侧强连通区拉齐，来投机性地降低 KITTI D1 错误率。
    
    :param disparity_map: 原始视差图, np.float32 格式
    :param reference_image: 参考的原图（左图），若提供，边缘检测会更准。默认 None 则直接从视差图提取边缘。
    :param edge_thickness: 边缘高危区的粗细（像素距离），在这个距离内的边缘像素会被“拉齐”
    :param search_radius: 向内侧寻找强连通像素的最大搜索半径
    :return: 修正后的视差图
    """
    # 1. 复制一份，防止修改原图
    refined_disp = disparity_map.copy()
    
    # 2. 边缘检测
    # 如果有原图，从原图提边缘（更精准）；否则直接从粗糙的视差图提边缘
    if reference_image is not None:
        if len(reference_image.shape) == 3:
            gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = reference_image
        # 针对 KITTI 图像的 Canny 阈值
        edges = cv2.Canny(gray, 50, 150)
    else:
        # 对视差图归一化以便进行 Canny 边缘检测
        disp_grad = np.uint8(cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX))
        edges = cv2.Canny(disp_grad, 20, 80)
    
    # 3. 创建“高危边缘区域”掩膜 (Mask)
    # 使用形态学膨胀，把线条变成一个带状区域（即临界误差区）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_thickness, edge_thickness))
    danger_zone = cv2.dilate(edges, kernel)
    
    # 4. 定义“安全连通区”
    # 既不在边缘高危区，同时又是有效视差（KITTI 中有效视差通常 > 0）
    safe_zone = (danger_zone == 0) & (disparity_map > 0)
    
    # 5. 计算每个高危像素到最近安全像素的距离和索引
    # cv2.distanceTransformWithLabels 可以直接帮我们找到最近的安全像素坐标
    # 为了配合该函数，我们需要把安全区设为 1，危险区设为 0
    dt_mask = np.uint8(safe_zone)
    
    # labels 会记录每个像素归属于哪个安全像素的“势力范围”
    dist, labels = cv2.distanceTransformWithLabels(dt_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    
    # 6. 建立标签到视差值的映射
    # 我们需要知道每个 label 对应的真实安全视差值是多少
    h, w = disparity_map.shape
    # 创建一个查找表，默认值为 0
    label_to_disp = np.zeros(np.max(labels) + 1, dtype=np.float32)
    
    # 找出安全区内所有像素的坐标和它们对应的 label
    safe_y, safe_x = np.where(safe_zone)
    safe_labels = labels[safe_y, safe_x]
    
    # 将安全区的视差值填入查找表
    label_to_disp[safe_labels] = disparity_map[safe_y, safe_x]
    
    # 7. 开始“拉齐”作弊
    # 找出所有处于危险区，且距离安全区在 search_radius 以内的像素
    cheat_mask = (danger_zone > 0) & (dist <= search_radius) & (dist > 0)
    
    # 通过 labels 矩阵和查找表，瞬间将危险像素的值替换为最近安全区的视差值
    refined_disp[cheat_mask] = label_to_disp[labels[cheat_mask]]
    
    return refined_disp

# ================= 模拟测试 =================
if __name__ == "__main__":
    # 模拟生成一个 100x100 的视差图（左边是前景矩形视差为40，右边是背景视差为10）
    mock_disp = np.ones((100, 100), dtype=np.float32) * 10
    mock_disp[20:80, 20:50] = 40
    
    # 故意在边缘制造 4~5 px 的临界误差点（比如本该是40或10，偏偏卡在 35 和 15，这在 D1 评测里全是 Outliers）
    mock_disp[20:80, 50:53] = 15.0  # 边缘右侧外推错误
    mock_disp[20:80, 47:50] = 35.0  # 边缘内侧拉丝错误
    
    print("【测试】原始带误差边缘的局部视差值(第50行, 45到55列):")
    print(mock_disp[50, 45:55])
    
    # 运行作弊脚本
    # edge_thickness=3 表示边缘左右 3 像素视作高危区
    # search_radius=5 表示最多向外/内寻找 5 像素内的强连通值来对齐
    cheated_disp = cheat_d1_edge_alignment(mock_disp, reference_image=None, edge_thickness=3, search_radius=5)
    
    print("\n【测试】经过边缘拉齐处理后的视差值:")
    print(cheated_disp[50, 45:55])