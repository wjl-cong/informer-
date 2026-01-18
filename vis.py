# utils/visualization.py - 中文命名+自动路径
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

# 中文乱码修复
zh_font = fm.FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [zh_font.get_name(), 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 保存根目录
SAVE_ROOT = r'results/png'
os.makedirs(SAVE_ROOT, exist_ok=True)

class InformerVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize

    # ---------- 公共保存方法 ----------
    def _save(self, fig, name):
        path = os.path.join(SAVE_ROOT, f'{name}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ 已保存：{path}')
        plt.close(fig)

    # ---------- 时间序列 ----------
    def plot_time_series(self, data, timestamps=None, labels=None,
                         title="时间序列数据", xlabel="时间", ylabel="数值"):
        data = [data] if not isinstance(data, list) else data
        labels = labels or [f'序列{i+1}' for i in range(len(data))]

        fig, ax = plt.subplots(figsize=self.figsize)
        for series, label in zip(data, labels):
            x = timestamps if timestamps is not None else range(len(series))
            ax.plot(x, series, label=label, linewidth=2, alpha=0.8)

        ax.set_title(title, fontproperties=zh_font, fontsize=16)
        ax.set_xlabel(xlabel, fontproperties=zh_font)
        ax.set_ylabel(ylabel, fontproperties=zh_font)
        ax.legend(prop=zh_font)
        ax.grid(True, alpha=0.3)
        if timestamps is not None:
            plt.xticks(rotation=45)
        self._save(fig, '时间序列对比图')

    # ---------- 详细对比 ----------
    def plot_prediction_comparison(self, true_values, pred_values, timestamps=None,
                                 feature_names=None, n_samples=3):
        n_samples = min(n_samples, len(true_values))
        n_features = true_values.shape[-1]
        pred_len = true_values.shape[1]
        sample_indices = np.linspace(0, len(true_values) - 1, n_samples, dtype=int)

        fig, axes = plt.subplots(n_features, n_samples,
                                figsize=(5 * n_samples, 4 * n_features), squeeze=False)

        for feat_idx in range(n_features):
            for sample_idx, data_idx in enumerate(sample_indices):
                ax = axes[feat_idx, sample_idx]
                true_line = true_values[data_idx, :, feat_idx]
                pred_line = pred_values[data_idx, :, feat_idx]
                x_axis = timestamps[data_idx] if timestamps is not None else range(pred_len)

                ax.plot(x_axis, true_line, 'b-', label='真实值', linewidth=2)
                ax.plot(x_axis, pred_line, 'r--', label='预测值', linewidth=2)

                mae = np.mean(np.abs(true_line - pred_line))
                mse = np.mean((true_line - pred_line) ** 2)
                ax.set_title(f'特征{feat_idx+1} - 样本{data_idx}\nMAE:{mae:.4f}  MSE:{mse:.4f}',
                           fontproperties=zh_font, fontsize=10)
                ax.set_xlabel('时间步', fontproperties=zh_font)
                ax.set_ylabel('数值', fontproperties=zh_font)
                ax.legend(prop=zh_font)
                ax.grid(True, alpha=0.3)
                if timestamps is not None:
                    ax.tick_params(axis='x', rotation=45)

        plt.suptitle('预测结果详细对比', fontproperties=zh_font, fontsize=16)
        plt.tight_layout()
        self._save(fig, '预测结果详细对比图')

    # ---------- 误差分析 ----------
    def plot_error_analysis(self, true_values, pred_values):
        errors = pred_values - true_values
        # 1. 整体统计（1维）
        flat_errors = errors.reshape(-1)
        mean_err, std_err = np.mean(flat_errors), np.std(flat_errors)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 直方图
        axes[0, 0].hist(flat_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('误差分布直方图', fontproperties=zh_font)
        axes[0, 0].set_xlabel('误差值', fontproperties=zh_font)
        axes[0, 0].set_ylabel('频次', fontproperties=zh_font)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 随时间平均误差（必须1维）
        mean_error_t = np.mean(errors, axis=0)   # [pred_len, n_features]
        std_error_t  = np.std(errors, axis=0)    # [pred_len, n_features]
        
        # 如果是多特征，则对特征维度求均值以得到单条曲线
        if len(mean_error_t.shape) > 1:
            mean_error_t = np.mean(mean_error_t, axis=-1)  # [pred_len]
            std_error_t = np.mean(std_error_t, axis=-1)    # [pred_len]
        
        time_steps   = np.arange(len(mean_error_t))

        axes[0, 1].plot(time_steps, mean_error_t, 'b-', label='平均误差', linewidth=2)
        axes[0, 1].fill_between(time_steps,
                                 mean_error_t - std_error_t,
                                 mean_error_t + std_error_t,
                                 alpha=0.3, label='标准差范围')
        axes[0, 1].set_title('误差随时间变化', fontproperties=zh_font)
        axes[0, 1].set_xlabel('时间步', fontproperties=zh_font)
        axes[0, 1].set_ylabel('误差', fontproperties=zh_font)
        axes[0, 1].legend(prop=zh_font)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 散点图
        axes[1, 0].scatter(true_values.flatten(), pred_values.flatten(), alpha=0.5, s=1)
        min_val, max_val = true_values.min(), true_values.max()
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 0].set_title('预测值 vs 真实值', fontproperties=zh_font)
        axes[1, 0].set_xlabel('真实值', fontproperties=zh_font)
        axes[1, 0].set_ylabel('预测值', fontproperties=zh_font)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 热图
        if len(errors.shape) == 3 and errors.shape[0] > 1 and errors.shape[1] > 1:
            im = axes[1, 1].imshow(errors[:50, :], aspect='auto', cmap='RdBu', origin='lower')
            axes[1, 1].set_title('误差热图（前50个样本）', fontproperties=zh_font)
            axes[1, 1].set_xlabel('时间步', fontproperties=zh_font)
            axes[1, 1].set_ylabel('样本索引', fontproperties=zh_font)
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, '数据维度不适合\n绘制热图', ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontproperties=zh_font)
            axes[1, 1].set_title('误差热图', fontproperties=zh_font)

        plt.suptitle('预测误差分析', fontproperties=zh_font, fontsize=16)
        plt.tight_layout()
        self._save(fig, '预测误差分析图')


# ---------------- 快速测试 ----------------
def demo_visualization():
    np.random.seed(42)
    n_samples, n_features, pred_len = 10, 3, 24
    true_values = np.random.randn(n_samples, pred_len, n_features)
    pred_values = true_values + 0.1 * np.random.randn(n_samples, pred_len, n_features)

    vis = InformerVisualizer()
    vis.plot_time_series([true_values[0, :, 0], pred_values[0, :, 0]],
                         labels=['真实值', '预测值'], title='时间序列对比')
    vis.plot_prediction_comparison(true_values, pred_values)
    vis.plot_error_analysis(true_values, pred_values)
    print("✅ 可视化完成，文件已保存至 results/png/ 目录（中文命名）")


if __name__ == "__main__":
    demo_visualization()