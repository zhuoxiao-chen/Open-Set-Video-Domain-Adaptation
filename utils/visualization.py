import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


import seaborn as sns
import torch
import numpy as np

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
import matplotlib.cm as cm
from matplotlib.lines import Line2D
def visualize_TSNE(source_feat, target_feat, source_label, target_label, path, class_names):

     # for open set open

     sns.set_style('darkgrid')
     sns.set_palette('muted')
     sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

     num_source = source_feat.size()[0]
     num_target = target_feat.size()[1]
     X = np.concatenate([source_feat.cpu().detach().numpy(), target_feat.cpu().detach().numpy()])
     # y_source = np.zeros((source_feat.size()[0], )) + 2
     # y_target = np.ones((target_feat.size()[0], )) * 5
     # y = np.concatenate([y_source, y_target])
     y = np.concatenate([source_label.detach().numpy(), target_label.detach().numpy()])

     digits_proj = TSNE(random_state=1).fit_transform(X)

     # We choose a color palette with seaborn.
     # palette = np.array(sns.color_palette("Paired"))
     # palette = plt.get_cmap('Set3')
     class_names = np.array(class_names)

     # We create a scatter plot.
     f = plt.figure(figsize=(6, 6))
     ax = plt.subplot(aspect='equal')
     # sc = ax.scatter(digits_proj[:num_source, 0], digits_proj[:num_source, 1], lw=0, s=40, marker='o',
     #                 c=palette[y[:num_source].astype(np.int)], alpha=0.5)
     # sc1 = ax.scatter(digits_proj[num_source:, 0], digits_proj[num_source:, 1], lw=0, s=40, marker='^',
     #                 c=palette[y[num_source:].astype(np.int)], alpha=0.5)
     index = y[:num_source].astype(np.int)

     # cmap_1.colors = cmap_1.colors[:-2]
     # cmap_1.N = 6
     # cmap_2 = plt.cm.Set2
     # cmap_2.colors = cmap_2.colors[:-1]
     # cmap_2.N = 7

     cmap_source = cm.get_cmap('Set2', 7)
     cmap_source.colors = cmap_source.colors[:-1]
     cmap_source._i_bad, cmap_source._i_over, cmap_source._i_under,  cmap_source.N = 8,7,6,6

     cmap_target = cm.get_cmap('Set2', 7)

     sc1 = ax.scatter(digits_proj[num_source:, 0], digits_proj[num_source:, 1], lw=0, s=40, marker='^',
                    c=y[num_source:], cmap=cmap_target, alpha=0.8)


     c = ax.scatter(digits_proj[:num_source, 0], digits_proj[:num_source, 1], lw=0, s=40, marker='o', c=y[:num_source], cmap=cmap_source, alpha=0.8)
     
     
     # customized = []
     # for i in range(len(class_names)):
     #     line = Line2D([0],[0],color=cm.Set3(i), label=class_names[i])
     #     customized.append(line)
     #
     # ax.legend(customized)
     # plt.xlim(-25, 25)
     # plt.ylim(-25, 25)
     ax.axis('off')
     ax.axis('tight')
     plt.savefig(path)

     txts = []

     # We add the labels for each digit.
     # txts = ["back_pack", "bike", "bike_helmet", "bookcase", "bottle",
     #                        "calculator", "desk_chair","desk_lamp","desktop_computer","file_cabinet","unk"]
     # for i in range(len(txts)):
     #     # Position of each label.
     #     xtext, ytext = np.median(digits_proj[y == i, :], axis=0)
     #     txt = ax.text(xtext, ytext, txts[i], fontsize=12)
     #     txt.set_path_effects([
     #         PathEffects.Stroke(linewidth=5, foreground="w"),
     #         PathEffects.Normal()])
     #     txts.append(txt)

