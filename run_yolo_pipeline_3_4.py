import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ================= 配置区域 =================
SPLITS_DIR = "./data_splits"       # 你的 JSON 分割表所在文件夹
YOLO_DATA_ROOT = "./YOLO_Data"     # 生成的 YOLO 数据集存放总目录
# ===========================================

def run_fold(fold):
    print(f"\n{'='*50}")
    print(f"🚀 开始全自动处理 Fold {fold}")
    print(f"{'='*50}\n")
    
    # ---------------------------------------------------------
    # 第一步：构建当前 Fold 的 YOLO 数据集
    # ---------------------------------------------------------
    yolo_dir = os.path.join(YOLO_DATA_ROOT, f"Fold_{fold}")
    os.makedirs(yolo_dir, exist_ok=True)
    
    dirs_to_make = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs_to_make:
        os.makedirs(os.path.join(yolo_dir, d), exist_ok=True)
        
    json_path = os.path.join(SPLITS_DIR, f"fold_{fold}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"📁 正在构建 Fold {fold} 数据集...")
    for split in ['train', 'val']:
        for item in tqdm(data[split], desc=f"复制 {split} 集"):
            img_path = item['image']
            # 替换路径寻找对应的 txt 标签
            lbl_path = img_path.replace('Original', 'YOLO_Label').replace(Path(img_path).suffix, '.txt')
            
            # 复制原图
            shutil.copy(img_path, os.path.join(yolo_dir, f"images/{split}", os.path.basename(img_path)))
            
            # 复制标签（如果不存在则创建空文件，YOLO 支持负样本）
            dst_lbl_path = os.path.join(yolo_dir, f"labels/{split}", os.path.basename(lbl_path))
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, dst_lbl_path)
            else:
                open(dst_lbl_path, 'w').close()
                
    yaml_path = os.path.join(yolo_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(yolo_dir)}\n")
        f.write("train: images/train\nval: images/val\n\nnames:\n  0: tear_meniscus\n")

    # ---------------------------------------------------------
    # 第二步：调用 YOLO 官方 API 自动炼丹
    # ---------------------------------------------------------
    print(f"\n🔥 开始训练 Fold {fold} 的 YOLO 模型...")
    model = YOLO('yolov8n.pt') 
    
    # 🔥🔥🔥 核心修改点：
    # 1. epochs 拉满到 150，榨干检测潜力
    # 2. 加入 exist_ok=True，防止新建 fold_x2 文件夹导致下面读不到权重
    # 🔥🔥🔥 核心修改点：
    # 1. 启动 8 卡 DDP 极速分布式训练
    # 2. 提升 batch 到 128 (甜点位)
    model.train(
        data=yaml_path, 
        epochs=150, 
        patience=35,             
        imgsz=1024, 
        batch=64,               # <--- 提速关键：扩大 Batch Size
        device=[0,1,2,3,4,5,6,7],# <--- 提速关键：唤醒 8 张 A40
        project='YOLO_Outputs',  
        name=f'fold_{fold}',     
        exist_ok=True,           
        verbose=False
    )
    
    # ---------------------------------------------------------
    # 第三步：加载刚训练好的最佳权重，进行推理预测
    # ---------------------------------------------------------
    best_pt = f"runs/detect/YOLO_Outputs/fold_{fold}/weights/best.pt"
    print(f"\n🎯 训练完成！加载权重进行推理: {best_pt}")
    infer_model = YOLO(best_pt)
    
    preds = {}
    print(f"🔍 正在生成 Fold {fold} 预测框...")
    for item in tqdm(data['val'], desc=f"Fold {fold} 推理"):
        img_path = item['image']
        img_id = item['id']
        
        res = infer_model(img_path, verbose=False)
        boxes = res[0].boxes
        
        if len(boxes) > 0:
            # 提取置信度最高的一个框的归一化坐标 [0~1]
            preds[img_id] = boxes.xyxyn[0].cpu().numpy().tolist()
        else:
            # 没检测到，给全图框
            preds[img_id] = [0.0, 0.0, 1.0, 1.0]
            
    out_json = os.path.join(SPLITS_DIR, f"yolo_boxes_fold{fold}.json")
    with open(out_json, 'w') as f:
        json.dump(preds, f, indent=4)
        
    print(f"\n✅ Fold {fold} 全部流程处理完毕！预测框已保存至: {out_json}")

if __name__ == '__main__':
    # 🔥🔥🔥 核心修改点：跑满全部 5 个 Fold
    for i in range(5):
        run_fold(i)
    
    print("\n🎉🎉🎉 恭喜！所有 5 个 Fold 的最新高精度 YOLO 预测框全部生成完毕！")