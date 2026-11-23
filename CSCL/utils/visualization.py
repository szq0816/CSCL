import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

colorList = [
    'darkblue',
    'darkviolet',
    'red',
    'deeppink',
    'darkslategray',
    'darkgreen',
    'fuchsia',
    'crimson',
    'aqua',
    'steelblue',
    'dodgerblue',
    'royalblue',
    'greenyellow',
    'lightblue',
    'lightgreen',
    'slateblue',
    'orange',
    'orangered',
    'yellow',
    'gold',
    'sandybrown',
    'gray',
]


def TSNE_show2D(z, y):
    """use t-SNE to visualize the latent representation"""
    t_sne = TSNE(n_components=2, learning_rate='auto')
    data = t_sne.fit_transform(z)
    data = pd.DataFrame(data, index=y)
    color = [colorList[i - 1] for i in data.index]

    plt.scatter(data[0], data[1], c=color, marker='.', s=12)
    plt.axis('off')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('../imgs/' + str(datetime.datetime.strftime(datetime.datetime.now(), '%d %H-%M-%S')) + '.pdf')
    plt.show()


def TSNE_show3D(z, y):
    """use t-SNE to visualize 3D the latent representation"""
    t_sne = TSNE(n_components=3, learning_rate='auto')
    data = t_sne.fit_transform(z)
    data = pd.DataFrame(data, index=y)
    color = [colorList[i - 1] for i in data.index]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], data[2], c=color, s=12)

    ax.set_xlim3d(min(data[0]), max(data[0]))
    ax.set_ylim3d(min(data[1]), max(data[1]))
    ax.set_zlim3d(min(data[2]), max(data[2]))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.tick_params(labelsize=8)
    # plt.axis('off')
    ax.grid(True)
    # ax.view_init(10, 185)
    plt.subplots_adjust(left=-0., right=1., top=1., bottom=-0.)
    plt.tight_layout()
    plt.savefig('../imgs/' + str(datetime.datetime.strftime(datetime.datetime.now(), '%d %H-%M-%S')) + '.pdf')
    plt.show()


def loss_plot(loss, acc, nmi, ari):
    epochs = range(1, len(loss) + 1)
    fig = plt.figure(figsize=(8, 5))
    ax_left = fig.add_subplot(111)
    ax_right = ax_left.twinx()

    ax_left.set_xlabel('Epoch', fontsize=14)
    ax_left.set_ylabel('Clustering Performance', fontsize=14)
    ax_right.set_ylabel('Loss', fontsize=14)

    a1 = ax_right.plot(epochs, loss, color='#F27970', label='Loss')
    a2 = ax_left.plot(epochs, acc, color='#54B345', label='ACC')
    a3 = ax_left.plot(epochs, nmi, color='#05B9E2', label='NMI')
    a4 = ax_left.plot(epochs, ari, color='#BB9727', label='ARI')

    lns = a1 + a2 + a3 + a4
    labs = [l.get_label() for l in lns]
    ax_left.legend(lns, labs, loc='center right')
    # ax_right.legend(loc='lower right', fontsize=10)
    # ax_left.legend(loc='center right', fontsize=10)

    plt.tight_layout()
    plt.savefig('../imgs/loss.pdf')
    plt.show()





def k_anl_plot():
    csv_dir='../data/'
    # 读取数据
    data = pd.read_csv(csv_dir+'K.csv')
    data = np.array(data)
    K = range(1, len(data[:, 0]) + 1)

    plt.figure(figsize=(8, 5))

    # 自定义颜色
    colors = ['#54B345', '#05B9E2', '#c82423']
    plt.plot(K, data[:, 0], color=colors[0], label='ACC', marker='o')
    plt.plot(K, data[:, 1], color=colors[1], label='NMI', marker='+')
    plt.plot(K, data[:, 2], color=colors[2], label='ARI', marker='*')

    plt.xlabel('Number of K', fontsize=16)
    plt.ylabel('Clustering Performance (%)', fontsize=16)
    # plt.title('ARI with different missing rates', fontsize=18)

    plt.legend(fontsize=10, loc='upper right')
    plt.tick_params(labelsize=12)
    plt.xticks(K[::2])

    # 保存为PDF格式并边距极小
    plt.tight_layout()
    plt.savefig('../imgs/K_anl.pdf')

