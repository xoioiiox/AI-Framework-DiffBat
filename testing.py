import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset as tfds
from gaussian_diffusion import GaussianDiffusion
from diffusion_model import DiffusionModel
from postproc_utils import PostProcess

# 测试脚本同样设置 GPU 显存按需增长。
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices(gpus, 'GPU')

# 选择要测试的预训练模型和数据集。
battery_dataset = 'mix'
total_timesteps = 1000
p_uncond = 0.2

# 加载训练好的 TensorFlow SavedModel。
mdir = './trained_models/'
network = keras.models.load_model(mdir+f'{battery_dataset}_network')
ema_network = keras.models.load_model(mdir+f'{battery_dataset}_ema_network')

# 构造与训练阶段一致的扩散工具。
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# 这里只用于生成/推理，不再 compile 和训练。
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
    p_uncond=p_uncond,
)

# 训练集这里被加载但未实际使用；测试评价主要用 test_ds。
train_ds = tfds.load(f'./data/{battery_dataset}_train_ds')
test_ds = tfds.load(f'./data/{battery_dataset}_test_ds')

# 后处理负责生成曲线、还原 SOH 百分比尺度，并计算评价指标。
post_process = PostProcess(test_ds, model)
refs, preds = post_process.pred(reps=1)
soh_rmse = post_process.eval_soh(refs, preds)
rul_rmse = post_process.eval_rul(refs, preds)

post_process.plot_sample(refs, preds)
print('RUL RMSE', rul_rmse, 'SOH RMSE', soh_rmse)
