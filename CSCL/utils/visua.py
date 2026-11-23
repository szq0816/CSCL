import datetime
import matplotlib.pyplot as plt
import numpy as np
import re

colorList = [
    'darkblue', 'darkviolet', 'red', 'deeppink', 'darkslategray', 'darkgreen',
    'fuchsia', 'crimson', 'aqua', 'steelblue', 'dodgerblue', 'royalblue',
    'greenyellow', 'lightblue', 'lightgreen', 'slateblue', 'orange', 'orangered',
    'yellow', 'gold', 'sandybrown', 'gray',
]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman']
plt.rcParams['axes.unicode_minus'] = False
def load_metrics_from_log(log_path, run_index=0):
    """
    读取日志，并返回指定训练 run_index 的 loss, acc, nmi, ari。
    run_index 从 0 开始
    """
    # 匹配每行指标
    pattern = re.compile(
        r"Epoch:\d+/\d+\s*\|\s*loss=([\d.]+)\s*\|\s*ACC=([\d.]+)\s*\|\s*NMI=([\d.]+)\s*\|\s*ARI=([\d.]+)"
    )

    # 先按训练段分组
    all_runs = []  # 每次训练的 metrics
    current_run = {'loss': [], 'acc': [], 'nmi': [], 'ari': []}

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # 解析指标
                loss_val = float(match.group(1))
                acc_val = float(match.group(2))
                nmi_val = float(match.group(3))
                ari_val = float(match.group(4))

                # 检查是否是新的训练 run
                if len(current_run['loss']) > 0 and int(re.search(r"Epoch:(\d+)/\d+", line).group(1)) == 1:
                    # 保存上一段训练
                    all_runs.append(current_run)
                    current_run = {'loss': [], 'acc': [], 'nmi': [], 'ari': []}

                current_run['loss'].append(loss_val)
                current_run['acc'].append(acc_val)
                current_run['nmi'].append(nmi_val)
                current_run['ari'].append(ari_val)

        # 保存最后一段训练
        if current_run['loss']:
            all_runs.append(current_run)

    if run_index >= len(all_runs):
        raise ValueError(f"日志里只有 {len(all_runs)} 次训练，run_index={run_index} 超出范围")

    run = all_runs[run_index]
    return run['loss'], run['acc'], run['nmi'], run['ari']

def loss_plot(loss, acc, nmi, ari, max_epoch=None, save_path='../imgs/handwritten_loss.pdf', run_index=None):
    if max_epoch is not None:
        loss = loss[:max_epoch]
        acc  = acc[:max_epoch]
        nmi  = nmi[:max_epoch]
        ari  = ari[:max_epoch]

    epochs = range(1, len(loss) + 1)

    fig = plt.figure(figsize=(8, 5))
    ax_left = fig.add_subplot(111)
    ax_right = ax_left.twinx()

    ax_left.set_xlabel('Epoch', fontsize=14)
    ax_left.set_ylabel('Clustering Performance', fontsize=14)
    ax_right.set_ylabel('Loss', fontsize=14)

    a1 = ax_right.plot(epochs, loss, color='red', label='Loss')
    a2 = ax_left.plot(epochs, acc, label='ACC')
    a3 = ax_left.plot(epochs, nmi, label='NMI')
    a4 = ax_left.plot(epochs, ari, label='ARI')

    lns = a1 + a2 + a3 + a4
    labs = [l.get_label() for l in lns]
    ax_left.legend(lns, labs, loc='center right')

    plt.tight_layout()

    # ===== 保存图像 =====
    if save_path is not None:
        if run_index is not None:
            # 在文件名里加 run_index
            dot_pos = save_path.rfind('.')
            save_path = save_path[:dot_pos] + f'_run{run_index}' + save_path[dot_pos:]
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")

    plt.show()



if __name__ == '__main__':
    log_file = '/home/zl/Code/CSCL/loss、acc/handwritten 00 2025-12-24 18-14-13.logs'
    run_index = 0
    loss, acc, nmi, ari = load_metrics_from_log(log_file, run_index=run_index)
    print(f'Loaded {len(loss)} epochs from run_index={run_index}.')
    loss_plot(loss, acc, nmi, ari)
