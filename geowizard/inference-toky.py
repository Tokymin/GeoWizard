import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from PIL import Image
# from Utils.calculate_depth_metrics import compute_errors
import matplotlib.pyplot as plt
import os
from config.config_test import hyperparameters as param


def compute_errors(gt_input, pred_input, tgt_img_input):
    # 确保 pred_input 是 (N, 320, 320, 1)，若已是正确维度则跳过
    if pred_input.ndim == 3:
        pred_input = np.expand_dims(pred_input, axis=-1)

    metrics = {
        'abs_diff': [], 'abs_rel': [], 'sq_rel': [], 'log10': [],
        'rmse': [], 'rmse_log': [], 'a1': [], 'a2': [], 'a3': []
    }
    max_depth = 200
    min_depth = 0.1

    for i in range(gt_input.shape[0]):
        gt = gt_input[i, ..., 0] if gt_input.shape[-1] > 1 else gt_input[i]  # 如果 gt 是多通道，取第一通道
        pred = pred_input[i, ..., 0]  # 从 (N, H, W, 1) 提取为 (H, W)
        tgt_img = tgt_img_input[i]  # 保持原维度

        # 创建有效掩码
        valid_mask = np.logical_and(gt > 0, gt <= max_depth)
        valid_gt = gt[valid_mask]
        valid_pred = pred[valid_mask]

        # 根据中值进行比例缩放和裁剪
        if valid_gt.size > 0 and valid_pred.size > 0:
            scale_factor = np.median(valid_gt) / (np.median(valid_pred) + 1e-6)
            valid_pred *= scale_factor
            valid_pred = np.clip(valid_pred, min_depth, max_depth)

            # 计算误差指标
            thresh = np.maximum(valid_gt / valid_pred, valid_pred / valid_gt)
            metrics['a1'].append(np.mean(thresh < 1.25))
            metrics['a2'].append(np.mean(thresh < 1.25 ** 2))
            metrics['a3'].append(np.mean(thresh < 1.25 ** 3))
            diff = valid_gt - valid_pred
            metrics['abs_diff'].append(np.mean(np.abs(diff)))
            metrics['abs_rel'].append(np.mean(np.abs(diff) / valid_gt))
            metrics['sq_rel'].append(np.mean((diff ** 2) / valid_gt))
            metrics['rmse'].append(np.sqrt(np.mean(diff ** 2)))
            metrics['rmse_log'].append(np.sqrt(np.mean((np.log(valid_gt) - np.log(valid_pred)) ** 2)))
            metrics['log10'].append(np.mean(np.abs(np.log10(valid_gt) - np.log10(valid_pred))))

    # 汇总结果
    results = {k: (np.mean(v), np.std(v)) for k, v in metrics.items() if v}
    return results


def is_nan_judge(value):
    return np.isnan(value)


def load_and_normalize_image(file_path):
    # 仅仅进行灰度值归一化发现gt 和pred的值还是有数量级差距
    """Load an image, convert to grayscale, normalize, and return as numpy array."""
    with Image.open(file_path) as img:
        img = (np.array(img.convert('L'), dtype=np.float32)) / 255  # / 255.0
        if img.shape != (320, 320):
            img = cv2.resize(img, (320, 320))
        return img


def load_and_normalize_image_serv(file_path):
    # 仅仅进行灰度值归一化发现gt 和pred的值还是有数量级差距
    """Load an image, convert to grayscale, normalize, and return as numpy array."""
    with Image.open(file_path) as img:
        original_img = np.array(img)
        normalized_img = (original_img - np.min(original_img)) / (np.max(original_img) - np.min(original_img))
        # img = np.array(img.convert('L'), dtype=np.float32) / 255.0 # / 255.0
        if normalized_img.shape != (320, 320):
            normalized_img = cv2.resize(normalized_img, (320, 320))
        return normalized_img


def load_image_color(file_path):
    """Load an image in full color."""
    with Image.open(file_path) as img:
        img = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
        if img.shape != (320, 320, 3):
            img = cv2.resize(img, (320, 320))
        return img


def load_matched_images(tgt_folder, gt_folder, pred_folder, start, num_images, prefix):
    """Load images based on filenames in the tgt_folder."""
    tgt_files = sorted([f for f in os.listdir(tgt_folder) if f.endswith(".png")])[start:num_images]
    tgt_images, gt_depth, pred_depth = [], [], []
    for index, tgt_file in zip(range(len(tgt_files)), tgt_files):
        base_name = tgt_file.split('.')[0]
        gt_file = f"aov_{base_name}.png"  # 必要时需要修改,针对Unity Endoslam
        pred_file = f"{base_name}{prefix}.png"  # 必要时需要修改
        tgt_image_path = os.path.join(tgt_folder, tgt_file)
        gt_image_path = os.path.join(gt_folder, gt_file)
        pred_image_path = os.path.join(pred_folder, pred_file)
        if os.path.exists(gt_image_path) and os.path.exists(pred_image_path) and os.path.exists(tgt_image_path):
            tgt_images.append(load_image_color(tgt_image_path))
            gt_depth.append(load_and_normalize_image_serv(gt_image_path))
            pred_depth.append(load_and_normalize_image(pred_image_path))
        elif not os.path.exists(gt_image_path):
            print(f"{gt_image_path}file does not exist")
        elif not os.path.exists(pred_image_path):
            print(f"{pred_image_path}file does not exist")
        elif not os.path.exists(tgt_image_path):
            print(f"{tgt_image_path}file does not exist")
    return np.array(tgt_images), np.array(gt_depth), np.array(pred_depth)


def run_script(tgt_img_folder, output_dir, num_images_to_load):
    import os
    os.makedirs(output_dir, exist_ok=True)
    import subprocess
    """python run_infer.py 
    --input_dir /mnt/share/toky/Projects/GeoWizard/test_Images/ 
    --output_dir /mnt/share/toky/Projects/GeoWizard/output 
    --ensemble_size 3 
    --denoise_steps 10 
    --seed 0 
    --domain "indoor"
    """
    # 配置参数
    domain = "indoor"
    inference_script = "run_infer.py"  # py 的路径
    ensemble_size = 3
    denoise_steps = 10
    seed = 0
    input_rgb_dir = tgt_img_folder
    output_dir = output_dir

    # 遍历目录下的所有 PNG 图像
    if not os.path.exists(tgt_img_folder):
        raise ValueError(f"Target image folder does not exist: {tgt_img_folder}")

    # png_files = sorted([f for f in os.listdir(tgt_img_folder) if f.endswith(".png")])

    tgt_files = sorted([f for f in os.listdir(tgt_img_folder) if f.endswith(".png")])[start:num_images_to_load]
    if not tgt_files:
        raise ValueError(f"No PNG files found in {tgt_img_folder}")
    # 遍历并调用 inference.py

    command = [
        "python",
        inference_script,
        "--domain", domain,
        "--ensemble_size", str(ensemble_size),
        "--seed", str(seed),
        "--denoise_steps", str(denoise_steps),
        "--input_dir", input_rgb_dir,
        "--output_dir", output_dir,
    ]
    try:
        # 执行命令
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 输出结果
        if result.returncode == 0:
            print(f"Success")
        else:
            print(f"Error processing: {result.stderr}")
    except Exception as e:
        print(f"Failed to process: {str(e)}")


def visualize_and_save_images(tgt_images, gt_images, pred_images, output_folder):
    """Create visualizations of images and save them."""
    os.makedirs(output_folder, exist_ok=True)
    for i in range(tgt_images.shape[0]):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(tgt_images[i], cmap='hot')
        plt.title('Target Image')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(gt_images[i], cmap='hot')
        plt.title('Ground Truth Depth')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(pred_images[i], cmap='hot')
        plt.title('Predicted Depth')
        plt.axis('off')
        plt.suptitle(f'Visualization {i}')
        plt.savefig(os.path.join(output_folder, f"vis_image_{i:04d}.png"))
        plt.close()


def save_results_to_excel(results, file_path, model_name):
    os.makedirs(file_path, exist_ok=True)
    data = {'Model Name': model_name}
    for metric, (mean_value, std_value) in results.items():
        data[f'{metric}_Mean'] = [mean_value]
        data[f'{metric}_Std'] = [std_value]  # 只保存均值
    df = pd.DataFrame(data)
    df.to_excel(file_path + "/results.xlsx", index=False)  # 保存到Excel
    print(f"Metrics saved to {file_path}")


if __name__ == '__main__':
    # step1: 加载图像
    start = param["start"]
    num_images_to_load = param["num_images_to_load"]  # Example number, change as needed
    is_visualize = param["is_visualize"]
    prefix = param["prefix"]  # 深度图存的前缀，例如depth_aov_
    input_folder = param['input_folder']
    gt_folder = param['gt_folder']
    save_metric_file_path = param["metrics_path"] + param["model_name"]
    output_folder = param["pred_depth_img_path"] + param["model_name"]
    vis_folder = param["vis_path"] + param["model_name"] + "/"
    # step2: 运行脚本
    if param["is_run_script"]:
        run_script(input_folder, output_folder, num_images_to_load)
        print("yes _run_script")
    # step3: 调用匹配函数找到GT
    tgt_images, gt_images, pred_images = load_matched_images(input_folder, gt_folder,
                                                             output_folder + "/depth_colored",
                                                             start, num_images_to_load, prefix)
    # step4: 计算可视化以及保存metric
    visualize_and_save_images(tgt_images, gt_images, pred_images, vis_folder)

    assert pred_images.shape[0] == gt_images.shape[0] == tgt_images.shape[0]
    # Compute metrics
    if param['is_save_metric']:
        results = compute_errors(gt_images, pred_images, tgt_images)
        save_results_to_excel(results, save_metric_file_path, param["model_name"])
    if param['is_visualize']:
        visualize_and_save_images(tgt_images, gt_images, pred_images, vis_folder)
