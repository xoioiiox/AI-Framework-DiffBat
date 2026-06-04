import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

class PostProcess():
    """生成结果的后处理与评价工具。

    模型直接生成长度 256 的归一化曲线；这里会插值到 2560 个 cycle，
    再换算回 SOH 百分比尺度，用于计算 SOH/RUL 指标。
    """

    def __init__(self, test_dataset, diffusion_model):
        super().__init__()
        # 评价时使用 1 到 2560 的 cycle 坐标。
        self.cycles = np.arange(0, 2560) + 1
        
        # 测试集原本是 batch 形式，unbatch 后逐条取出参考 SOH 和条件矩阵。
        test_ds = test_dataset.unbatch()

        self.reference_soh = np.array([x for x, _, c in test_ds])
        self.capacity_matrix = np.array([c for x, _, c in test_ds])
        self.N = len(self.reference_soh)
            
        # 已训练好的 DiffusionModel，用于根据条件矩阵生成曲线。
        self.model = diffusion_model

    def gen(self, guide_w=0.0, reps=1):
        # reps 表示每个条件矩阵重复生成几条候选曲线。
        protos = np.repeat(self.capacity_matrix, reps, axis = 0)
        return self.model.generate_samples(protos, guide_w=guide_w)
    
    def pred(self, guide_w=0.0, reps=1):
        # 参考曲线从长度 256 resize 到 2560，并从 [-1, 1] 还原到 SOH 百分比。
        refs = tf.image.resize(self.reference_soh[:, :, None, :], [2560, 1]).numpy()
        refs = (refs.reshape((self.N, -1)) + 1.0) / 2 * 100
        
        # 生成曲线同样 resize 到 2560 并还原尺度。
        samples = self.gen(guide_w, reps=reps)
        samples = tf.image.resize(samples[:, :, None, :], [2560, 1]).numpy()
        
        samples = (samples.reshape((self.N, reps, -1)) + 1.0) / 2 * 100
        
        # 如果每个样本生成多个候选曲线，使用前 100 cycle 的 SOH RMSE 选最优候选。
        rmse_100 = []
        for i in range(reps):
            soh_rmse = self.soh_rmse(refs[:, :100], samples[:, i, :100], 80)
            rmse_100.append(soh_rmse)
            
        rmse_100 = np.array(rmse_100).T
        indx = np.argmin(rmse_100, 1)
    
        preds = []
        for i in range(self.N):
            preds.append(samples[i, indx[i]])
            
        preds = np.array(preds)
        return refs, preds
    
    def soh_rmse(self, ref, pred, eol):
        # 低于阈值的片段不参与 SOH RMSE 计算。
        refc = ref.copy()
        refc[refc <= 60] = np.nan
        predc = pred.copy()
        predc[predc <= eol] = np.nan
        
        error = refc - predc
        soh_rmse = np.sqrt(np.nanmean(error**2, axis = 1))
        return soh_rmse
    
    def eval_soh(self, refs, preds, eol=80):
        # 所有测试样本的 SOH RMSE 均值。
        soh_rmse = self.soh_rmse(refs, preds, eol)
        return soh_rmse.mean()
    
    def get_rul(self, data, eol=80):
        # RUL 定义为 SOH 首次跌破/到达 EOL 阈值附近的 cycle。
        # 这里沿用原代码逻辑：寻找 data > eol 的边界索引。
        rul_index = np.argmin(data > eol, axis=1)
        has_crossing = rul_index != 0
        rul_index = np.where(has_crossing, rul_index, -1)
        rul = self.cycles[rul_index]
        return rul
        
    def eval_rul(self, refs, preds, eol = 80):
        # 分别计算参考曲线和预测曲线的 RUL，再计算 RMSE。
        indx = refs == 0.0
        refs[indx] = None
        
        rul_ref = self.get_rul(refs, eol=eol)
        rul_pred = self.get_rul(preds, eol=eol)
        
        rul_rmse = np.sqrt(np.mean((rul_ref - rul_pred)**2))
        
        return rul_rmse
    
    def plot_sample(self, ref, pred):
        # 将每条参考曲线和预测曲线画在同一子图中，便于直观比较。
        n = int(np.ceil(np.sqrt(ref.shape[0])))
        ref[ref <= 60] = np.nan
        pred[pred <= 60] = np.nan
        
        fig, ax = plt.subplots(n, n, figsize=(n/2, n/2), sharey=True)
        for i in range(ref.shape[0]):
            axx = ax.flatten()[i]

            axx.plot(self.cycles, ref[i], c='cyan', lw=1.5, label='Reference')
            axx.plot(self.cycles, pred[i], c='magenta', lw=1.5, ls = '--', label='Prediction')

            axx.set_ylim(80, 105)
            axx.set_xlim(0, self.cycles[max(np.argmin(ref[i]>80), np.argmin(pred[i]>80))] // 500 * 500 + 500)
            
            axx.set_xticks([])
            axx.grid()
            
        for i in range(n**2 - ref.shape[0]):
            axx = ax.flatten()[-i-1]
            axx.set_axis_off()
        
        for axx in ax[:1, 0]:
            axx.set_ylabel('SOH(%)', fontsize=8)
            
        for axx in ax[-1, :1]:
            axx.set_xlabel('Cycle', fontsize=8)
        plt.show()
