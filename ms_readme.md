# MindSpore版本复现

新增文件：

- ms_unet.py：MindSpore 版 DiffBatt U-Net，保持输入输出形状 [batch, 256, 1]
- ms_gaussian_diffusion.py：DDPM 加噪/去噪公式
- ms_diffusion_model.py：训练 loss cell、EMA 更新、采样器
- ms_data.py：把 TensorFlow snapshot 导出为 .npz，并用 MindSpore GeneratorDataset 读取
- ms_training.py：MindSpore 训练脚本
- ms_testing.py：MindSpore 测试/评估脚本
- ms_postproc_utils.py：不用 TensorFlow 的 SOH/RUL 后处理
- requirements_mindspore.txt：MindSpore 版依赖

```
venv\Scripts\python.exe ms_data.py --snapshot data/matr_1_train_ds --output data_npz/matr_1_train.npz
venv\Scripts\python.exe ms_data.py --snapshot data/matr_1_test_ds --output data_npz/matr_1_test.npz
venv\Scripts\python.exe ms_training.py --train-npz data_npz/matr_1_train.npz --epochs 1 --device CPU
venv\Scripts\python.exe ms_testing.py --test-npz data_npz/matr_1_test.npz --ema-ckpt checkpoints_ms/matr_1/ema_network.ckpt --device CPU
```



推荐配置：

```
# 训练
venv\Scripts\python.exe ms_training.py --train-npz data_npz/matr_1_train.npz --epochs 1000 --batch-size 41 --learning-rate 1e-3 --timesteps 1000 --p-uncond 0.2 --ema 0.999 --device CPU

# 测试
# 低配
venv\Scripts\python.exe ms_eval_metrics.py --test-npz data_npz/matr_1_test.npz --ema-ckpt checkpoints_ms/matr_1 --device CPU --timesteps 1000 --reps 3 --max-samples 8 --progress-every 50 --all-eol

venv\Scripts\python.exe ms_eval_metrics.py --test-npz data_npz/matr_1_test.npz --ema-ckpt checkpoints_ms/matr_1 --device CPU --timesteps 1000 --reps 10 --progress-every 50 --all-eol

# 绘图
venv\Scripts\python.exe ms_generate_figures.py --test-npz data_npz/matr_1_test.npz --ema-ckpt checkpoints_ms/matr_1 --device CPU --timesteps 1000 --reps 3 --max-samples 8 --max-panels 8 --progress-every 50 --output-dir figures_ms
```



对，论文 Figure 2 里有两条信息：

- **粉色线**：当前 denoising step 的生成样本，也就是从高斯噪声逐步去噪后的中间状态。
- **青色线**：该测试样本的真实 SOH 曲线，用作参考对照。

我已经把我们的 ms_generate_figures.py 改成论文这种画法了：每个小图都会叠加青色真实 SOH 线。

现在图的含义是：

- **x 轴**：Cycle number，循环次数。图上隐藏了刻度，但内部是从约 1 到 2560。
- **y 轴**：SOH(%)。
- **t = 0**：初始纯高斯噪声。
- **t = 100, 200, ... 1000**：反向扩散进行到对应步数后的样本。
- **粉色越接近青色**：说明去噪生成曲线越接近真实退化曲线。

```
venv\Scripts\python.exe ms_generate_figures.py --test-npz data_npz/matr_1_test.npz --ema-ckpt checkpoints_ms/matr_1 --device CPU --timesteps 1000 --max-samples 1 --max-panels 1 --progress-every 50 --denoise-checkpoints 100,200,300,400,500,600,700,800,900,1000 --output-dir figures_ms
```





```
venv\Scripts\python.exe ms_training.py --train-npz data_npz/matr_1_train.npz --val-npz data_npz/matr_1_test.npz --epochs 500 --batch-size 41 --learning-rate 1e-3 --timesteps 1000 --p-uncond 0.2 --ema 0.999 --device CPU --ckpt-dir checkpoints_ms/matr_1_v2

venv\Scripts\python.exe ms_eval_metrics.py --test-npz data_npz/matr_1_test.npz --ema-ckpt checkpoints_ms/matr_1_v2 --device CPU --timesteps 1000 --reps 3 --progress-every 50 --all-eol

venv\Scripts\python.exe ms_generate_figures.py --test-npz data_npz/matr_1_test.npz --ema-ckpt checkpoints_ms/matr_1_v2 --device CPU --timesteps 1000 --reps 3 --max-samples 8 --max-panels 8 --progress-every 50 --output-dir figures_ms_v2

venv\Scripts\python.exe ms_generate_figures.py --test-npz data_npz/matr_1_test.npz --ema-ckpt checkpoints_ms/matr_1_v2 --device CPU --timesteps 1000 --max-samples 1 --max-panels 1 --progress-every 50 --denoise-checkpoints 100,200,300,400,500,600,700,800,900,1000 --denoise-mode x0 --output-dir figures_ms_v2
```



python ms_training.py --train-npz data_npz/matr_1_train.npz --val-npz data_npz/matr_1_test.npz --epochs 500 --batch-size 41 --learning-rate 5e-4 --clip-norm 1.0 --timesteps 1000 --p-uncond 0.2 --ema 0.999 --device GPU --ckpt-dir checkpoints_ms/matr_1_v2_gpu --log-file logs_ms/train_matr1_v2_gpu.txt --log-every 1



python ms_eval_metrics.py --test-npz data_npz/matr_1_test.npz --ema-ckpt checkpoints_ms/matr_1_v2_gpu --device GPU --timesteps 1000 --reps 3 --progress-every 100 --all-eol --debug --clip-denoised --log-file logs_ms/eval_matr1_v2_gpu_reps3.txt



python ms_generate_figures.py --test-npz data_npz/matr_1_test.npz --ema-ckpt checkpoints_ms/matr_1_v2_quick --device GPU --timesteps 1000 --reps 1 --max-samples 1 --max-panels 8 --progress-every 100 --denoise-checkpoints 100,200,300,400,500,600,700,800,900,1000 --denoise-mode x0 --clip-denoised --output-dir figures_ms