import json
import math

def prune_trajectory_global(points, distance_threshold):
    """
    全局轨迹点剪枝（纯 Python）
    
    参数：
        points: list of dict，每个点至少包含 'pointX' 和 'pointY'
        distance_threshold: float，小于此距离的点会被剔掉
    
    返回：
        pruned_points: list of dict，保留的轨迹点
    """
    if not points:
        return []

    def dist(p1, p2):
        return math.sqrt((float(p1['pointX']) - float(p2['pointX']))**2 +
                         (float(p1['pointY']) - float(p2['pointY']))**2)

    keep_flags = [True] * len(points)

    for i, p1 in enumerate(points):
        if not keep_flags[i]:
            continue
        for j, p2 in enumerate(points):
            if i == j or not keep_flags[j]:
                continue
            if dist(p1, p2) < distance_threshold:
                keep_flags[j] = False

    pruned_points = [points[i] for i in range(len(points)) if keep_flags[i]]
    return pruned_points


# ==========================
# 使用示例
# ==========================
if __name__ == "__main__":
    # 读取文件
    with open("/home/ubuntu/Risk-project/dot/点坐标序.txt", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取 records
    trajectory = data.get("data", {}).get("records", [])

    print(f"原始点数: {len(trajectory)}")

    DIST_THRESHOLD = 300.0  # 设置距离阈值
    pruned = prune_trajectory_global(trajectory, DIST_THRESHOLD)

    # 保存剪枝后的结果
    with open("pruned_trajectory_final.json", "w", encoding="utf-8") as f:
        json.dump(pruned, f, indent=2)

    print(f"剪枝后点数: {len(pruned)}")