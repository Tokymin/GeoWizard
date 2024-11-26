import subprocess
import time
import numpy as np
import argparse


def run_script(script_name, args):
    """运行指定的脚本并传递参数，返回执行时间"""
    start_time = time.time()
    # 组装命令，将脚本名和参数拼接成一个列表
    command = ['python', script_name]
    command.extend([
        '--input_dir', args.input_dir,
        '--output_dir', args.output_dir,
        '--ensemble_size', str(args.ensemble_size),
        '--denoise_steps', str(args.denoise_steps),
        '--seed', str(args.seed),
        '--domain', args.domain
    ])

    subprocess.run(command, check=True)
    return time.time() - start_time


# 解析输入参数
parser = argparse.ArgumentParser(description="Compare inference time between original and optimized scripts.")
parser.add_argument('--input_dir', type=str, required=True, help="Input directory for images.")
parser.add_argument('--output_dir', type=str, required=True, help="Output directory for results.")
parser.add_argument('--ensemble_size', type=int, default=3, help="Number of predictions to ensemble.")
parser.add_argument('--denoise_steps', type=int, default=10, help="Number of denoising steps.")
parser.add_argument('--seed', type=int, default=0, help="Random seed.")
parser.add_argument('--domain', type=str, default="indoor", help="Domain type, e.g., 'indoor'.")
parser.add_argument('--num_runs', type=int, default=1, help="Number of runs to average the time.")
args = parser.parse_args()

# 记录每个脚本的推理时间
times_without_optimization = []
times_with_optimization = []

print("Running original script (without optimization)...")
for i in range(args.num_runs):
    elapsed_time = run_script("run_infer.py", args)
    times_without_optimization.append(elapsed_time)
    print(f"Run {i + 1}/{args.num_runs} - Time: {elapsed_time:.4f} seconds")

print("\nRunning optimized script (with optimization)...")
for i in range(args.num_runs):
    elapsed_time = run_script("acc_run_infer.py", args)
    times_with_optimization.append(elapsed_time)
    print(f"Run {i + 1}/{args.num_runs} - Time: {elapsed_time:.4f} seconds")

# 计算平均推理时间
avg_time_without_optimization = np.mean(times_without_optimization)
avg_time_with_optimization = np.mean(times_with_optimization)

# 输出对比结果
print("\nInference Time Comparison:")
print(f"Average inference time without optimization: {avg_time_without_optimization:.4f} seconds")
print(f"Average inference time with optimization: {avg_time_with_optimization:.4f} seconds")
speedup = avg_time_without_optimization / avg_time_with_optimization
print(f"Speedup factor: {speedup:.2f}x")
