import json
import math
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List

app = FastAPI()


# =========================
# 核心剪枝函数（你的原始逻辑）
# =========================
def prune_trajectory_global(points, distance_threshold):
    if not points:
        return []

    def dist(p1, p2):
        return math.sqrt(
            (float(p1['pointX']) - float(p2['pointX'])) ** 2 +
            (float(p1['pointY']) - float(p2['pointY'])) ** 2
        )

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


# =========================
# 请求格式定义
# =========================
class PruneRequest(BaseModel):
    data: Dict[str, Any]
    distance: float


# =========================
# 接口
# =========================
@app.post("/prune")
def prune_api(req: PruneRequest):
    try:
        # 解析轨迹
        trajectory = req.data.get("data", {}).get("records", [])

        if not isinstance(trajectory, list):
            return {"error": "records 必须是 list"}

        original_len = len(trajectory)

        # 执行剪枝
        pruned = prune_trajectory_global(trajectory, req.distance)

        # 保存结果
        output_file = "pruned_trajectory_final.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pruned, f, ensure_ascii=False, indent=2)

        return {
            "message": "success",
            "original_count": original_len,
            "pruned_count": len(pruned),
            "output_file": output_file,
            "data": pruned
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# 启动
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8071)