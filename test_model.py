from ultralytics import YOLO
import cv2
import os
from collections import defaultdict

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # box格式: [x1, y1, x2, y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0


def read_gt_boxes(label_path, img_width, img_height):
    """从YOLO格式的txt文件读取真实框，并转换为[x1,y1,x2,y2]格式"""
    gt_boxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        
        # 转换为绝对坐标
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        # 转换为[x1,y1,x2,y2]格式
        x1 = x_center_abs - width_abs / 2
        y1 = y_center_abs - height_abs / 2
        x2 = x1 + width_abs
        y2 = y1 + height_abs
        
        gt_boxes.append({
            'class_id': int(class_id),
            'bbox': [x1, y1, x2, y2]
        })
    
    return gt_boxes


def evaluate_image(model, image_path, label_dir):
    """评估单张图片（支持多类别）"""
    # 获取对应的标签文件路径
    image_name = os.path.basename(image_path)
    label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
    
    # 读取真实框
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    gt_boxes = read_gt_boxes(label_path, img_width, img_height) if os.path.exists(label_path) else []
    
    # 运行模型预测
    results = model(image_path)
    pred_boxes = []
    
    for result in results:
        for box in result.boxes:
            # 获取预测框坐标 (xyxy格式) 和类别ID
            bbox = box.xyxy[0].cpu().numpy().tolist()
            class_id = int(box.cls[0].cpu().numpy())
            pred_boxes.append({
                'class_id': class_id,
                'bbox': bbox
            })
    
    # 按类别组织真实框和预测框
    gt_by_class = defaultdict(list)
    for gt in gt_boxes:
        gt_by_class[gt['class_id']].append(gt['bbox'])
    
    pred_by_class = defaultdict(list)
    for pred in pred_boxes:
        pred_by_class[pred['class_id']].append(pred['bbox'])
    
    # 计算每个类别的指标
    class_metrics = {}
    total_gt = 0
    total_pred = 0
    
    for class_id in set(list(gt_by_class.keys()) + list(pred_by_class.keys())):
        class_gt = gt_by_class.get(class_id, [])
        class_pred = pred_by_class.get(class_id, [])
        
        true_positives = 0
        matched_gt_indices = set()
        
        for pred_box in class_pred:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(class_gt):
                if gt_idx in matched_gt_indices:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou > 0.65:  # IoU阈值
                true_positives += 1
                matched_gt_indices.add(best_gt_idx)
        
        precision = true_positives / len(class_pred) if class_pred else 0
        recall = true_positives / len(class_gt) if class_gt else 0
        
        class_metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'gt_count': len(class_gt),
            'pred_count': len(class_pred)
        }
        
        total_gt += len(class_gt)
        total_pred += len(class_pred)
    
    # 计算总体指标（所有类别的平均）
    total_precision = sum(m['precision'] * m['pred_count'] for m in class_metrics.values()) / total_pred if total_pred else 0
    total_recall = sum(m['recall'] * m['gt_count'] for m in class_metrics.values()) / total_gt if total_gt else 0
    
    return class_metrics, total_precision, total_recall, total_gt, total_pred

# 使用示例
model = YOLO("best.pt")
image_dir = "train_data/images/val"
label_dir = "train_data/labels/val"

all_class_metrics = []
total_precision = 0
total_recall = 0
total_gt_boxes = 0
total_pred_boxes = 0
image_count = 0

for image_file in os.listdir(image_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_file)
        class_metrics, precision, recall, gt_count, pred_count = evaluate_image(model, image_path, label_dir)
        
        all_class_metrics.append(class_metrics)
        total_precision += precision
        total_recall += recall
        total_gt_boxes += gt_count
        total_pred_boxes += pred_count
        image_count += 1
        
        print(f"\nProcessed {image_file}:")
        print(f"  GT Boxes: {gt_count}, Pred Boxes: {pred_count}")
        for class_id, metrics in class_metrics.items():
            print(f"  Class {class_id}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f} (GT:{metrics['gt_count']}, Pred:{metrics['pred_count']})")

if image_count > 0:
    avg_precision = total_precision / image_count
    avg_recall = total_recall / image_count
    
    print("\nFinal Results:")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Total Ground Truth Boxes: {total_gt_boxes}")
    print(f"Total Predicted Boxes: {total_pred_boxes}")
    
    # 可以进一步计算每个类别的总体指标
    final_class_metrics = defaultdict(lambda: {'precision': 0, 'recall': 0, 'gt_count': 0, 'pred_count': 0})
    for metrics in all_class_metrics:
        for class_id, m in metrics.items():
            final_class_metrics[class_id]['precision'] += m['precision'] * m['pred_count']
            final_class_metrics[class_id]['recall'] += m['recall'] * m['gt_count']
            final_class_metrics[class_id]['gt_count'] += m['gt_count']
            final_class_metrics[class_id]['pred_count'] += m['pred_count']
    
    print("\nPer-Class Final Metrics:")
    for class_id, m in final_class_metrics.items():
        avg_p = m['precision'] / m['pred_count'] if m['pred_count'] else 0
        avg_r = m['recall'] / m['gt_count'] if m['gt_count'] else 0
        print(f"Class {class_id}: Precision={avg_p:.4f}, Recall={avg_r:.4f} (GT:{m['gt_count']}, Pred:{m['pred_count']})")
else:
    print("No images found in the directory!")


