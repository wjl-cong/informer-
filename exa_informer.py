import os
import sys
import argparse
import torch
import numpy as np
import time
import warnings
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Informer2020'))
# =======================================================
warnings.filterwarnings('ignore')

# 现在可以正常导入官方模块
from Informer2020.exp.exp_informer import Exp_Informer
from Informer2020.utils.tools import EarlyStopping, adjust_learning_rate
from Informer2020.data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from Informer2020.models.model import Informer, InformerStack
from Informer2020.utils.metrics import metric
from Informer2020.utils.timefeatures import time_features

# -------------------- R² 指标（官方默认用 MSE，这里我是喜欢无脑用R方） --------------------
def calc_r2(y_true, y_pred):
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


# -------------------- 封装 --------------------
class Exp_Informer_Fixed(Exp_Informer):
    """
    继承官方 Exp_Informer，仅重写 train/test 以便加入 R² 与进度条
    """
    def train(self, setting):
        # 调用父类训练前先打印配置
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader   = self._get_data(flag='val')
        test_data, test_loader   = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_steps = len(train_loader)
        print(f'\n🚀 开始训练，总轮数: {self.args.train_epochs}')
        print(f'📊 训练步数每轮: {train_steps}')
        print(f'📦 批次大小: {self.args.batch_size}')
        print('=' * 80)

        epoch_bar = self._create_progress_bar(self.args.train_epochs, desc='🔄 训练进度', unit='轮')
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss, train_r2 = [], []

            self.model.train()
            epoch_time = time.time()
            batch_bar = self._create_progress_bar(train_steps, desc=f'📝 轮次 {epoch+1}/{self.args.train_epochs}', unit='批')

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                loss = criterion(pred, true)
                train_loss.append(loss.item())
                train_r2.append(calc_r2(true.detach().cpu().numpy(), pred.detach().cpu().numpy()))

                loss.backward()
                model_optim.step()

                batch_bar.update()
                batch_bar.set_postfix({'损失': f'{loss.item():.6f}', 'R²': f'{train_r2[-1]:.4f}'})

            batch_bar.close()

            train_loss = np.average(train_loss)
            train_r2 = np.average(train_r2)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            epoch_bar.update()
            epoch_bar.set_postfix({'训练损失': f'{train_loss:.6f}',
                                   '验证损失': f'{vali_loss:.6f}',
                                   '训练R²': f'{train_r2:.4f}'})

            print(f'\n📈 轮次 {epoch + 1} 完成 | 训练损失: {train_loss:.7f} | '
                  f'验证损失: {vali_loss:.7f} | 训练R²: {train_r2:.4f}')

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print('⏰ 触发早停，结束训练')
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        epoch_bar.close()

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        print(f'\n✅ 训练完成！')
        return self.model

    # -------------------- 测试 --------------------
    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()
        preds, trues = [], []

        print(f'\n🔍 开始测试，测试集大小: {len(test_data)}')
        bar = self._create_progress_bar(len(test_loader), desc='🔬 测试进度', unit='批')

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                bar.update()
        bar.close()

        preds = np.array(preds)
        trues = np.array(trues)
        print(f'📊 测试形状: {preds.shape}, {trues.shape}')
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print(f'📊 重塑后形状: {preds.shape}, {trues.shape}')

        # 计算指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        r2 = calc_r2(trues, preds)
        print('\n' + '=' * 60)
        print('📊 测试集评估结果:')
        print('=' * 60)
        print(f'🎯 R²决定系数: {r2:.6f}')
        print(f'📏 MAE: {mae:.6f}')
        print(f'📈 MSE: {mse:.6f}')
        print(f'📐 RMSE: {rmse:.6f}')
        print(f'📊 MAPE: {mape:.2f}%')
        print('=' * 60)

        # 保存
        folder = f'./results/{setting}/'
        os.makedirs(folder, exist_ok=True)
        np.save(folder + 'metrics.npy', np.array([r2, mae, mse, rmse, mape]))
        np.save(folder + 'pred.npy', preds)
        np.save(folder + 'true.npy', trues)

        return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}

    # -------------------- 进度条工具 --------------------
    def _create_progress_bar(self, total, desc='', unit='it'):
        try:
            from tqdm import tqdm
            return tqdm(total=total, desc=desc, unit=unit)
        except ImportError:
            class SimpleBar:
                def __init__(self, total, desc, unit):
                    self.total, self.desc, self.unit = total, desc, unit
                    self.current = 0
                def update(self, n=1):
                    self.current += n
                    p = self.current / self.total
                    bar = '█' * int(40 * p) + '-' * (40 - int(40 * p))
                    sys.stdout.write(f'\r{self.desc} |{bar}| {int(p*100)}% {self.current}/{self.total}')
                    sys.stdout.flush()
                def close(self): print()
                def set_postfix(self, **kw): print(' | '.join([f'{k} {v}' for k, v in kw.items()]), end='')
            return SimpleBar(total, desc, unit)


# -------------------- 参数配置 --------------------
def get_args():
    parser = argparse.ArgumentParser(description='Informer2020 官方完整参数版')
    # ========== 基本 ==========
    parser.add_argument('--model', type=str, default='informer')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./ETDataset/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--des', type=str, default='test', help='实验描述')

    # ========== 序列长度 ==========
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)

    # ========== 模型结构 ==========
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--factor', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--attn', type=str, default='prob')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--s_layers', type=str, default='3,2,1')
    parser.add_argument('--mix', action='store_true', default=True)

    # ========== 训练 ==========
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_epochs', type=int, default=30)          # 30轮
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=999)             # 禁用早停

    # ========== GPU ==========
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    # ========== 其它 ==========
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--do_predict', action='store_true', default=False)
    parser.add_argument('--output_attention', action='store_true', default=False)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--plot_results', action='store_true', default=True)

    # ========== 官方隐藏字段 ==========
    parser.add_argument('--use_amp', action='store_true', default=False, help='是否使用自动混合精度')
    parser.add_argument('--padding', type=int, default=0, help='decoder 输入填充方式')
    parser.add_argument('--inverse', action='store_true', default=False, help='是否反归一化')
    parser.add_argument('--tqdm', type=int, default=1, help='是否使用 tqdm')
    parser.add_argument('--cols', type=str, default=None, help='指定自定义列')
    parser.add_argument('--feature_size', type=int, default=0)
    parser.add_argument('--step_len', type=int, default=0)
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--data_name', type=str, default='ETTh1')
    parser.add_argument('--save_path', type=str, default='./')

    args = parser.parse_args()

    # 后处理
    args.features = args.features[-1:] if args.features else 'M'
    args.s_layers = [int(s) for s in args.s_layers.replace(' ', '').split(',')]
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # 数据维度映射
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
        'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
        'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
    }
    if args.data in data_parser:
        info = data_parser[args.data]
        args.data_path = info['data']
        args.target = info['T']
        args.enc_in, args.dec_in, args.c_out = info[args.features]

    print('\n' + '=' * 60)
    print('🚀 Informer2020 官方原版修复 – 配置如下')
    print('=' * 60)
    print(f'📊 数据集: {args.data}')
    print(f'🔧 模型: {args.model}')
    print(f'📏 输入长度: {args.seq_len}')
    print(f'🎯 预测长度: {args.pred_len}')
    print(f'🔄 训练轮数: {args.train_epochs}')
    print(f'📦 批次大小: {args.batch_size}')
    print(f'💻 使用设备: {"GPU" if args.use_gpu else "CPU"}')
    print('=' * 60 + '\n')

    return args

# -------------------- 主入口 --------------------
def main():
    args = get_args()
    exp = Exp_Informer_Fixed(args)

    setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fb{}_eb{}_dt{}_{}_{}'.format(
        args.model, args.data, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
        args.attn, args.factor, args.embed, args.distil, args.learning_rate, args.loss, args.des)

    # 训练
    print('🎯 开始训练...')
    exp.train(setting)

    # 测试
    print('\n🔍 开始测试...')
    exp.test(setting)

    # 预测
    if args.do_predict:
        print('\n🔮 开始预测...')
        exp.predict(setting, True)

    print('\n✅ 全部任务完成！')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()