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

