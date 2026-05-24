import argparse
import os
from struct import unpack
import gzip

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


def load_test():
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs / test_imgs.max()
    return test_imgs, test_labs


def confusion_matrix(labels, preds, num_classes=10):
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(labels, preds):
        mat[y, p] += 1
    return mat


def plot_confusion_matrix(mat, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_misclassified(images, labels, preds, save_path, max_count=16):
    wrong_idx = np.where(labels != preds)[0][:max_count]
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    axes = axes.reshape(-1)
    for ax_idx, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        if ax_idx < len(wrong_idx):
            idx = wrong_idx[ax_idx]
            ax.imshow(images[idx].reshape(28, 28), cmap='gray')
            ax.set_title(f'T:{labels[idx]} P:{preds[idx]}', fontsize=9)
        else:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_cnn_kernels(model, save_path):
    kernels = model.layers[0].params['W']
    count = kernels.shape[0]
    fig, axes = plt.subplots(1, count, figsize=(count * 1.3, 1.5))
    if count == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(kernels[i, 0], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'K{i}', fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_mlp_weights(model, save_path, max_count=16):
    weights = model.layers[0].params['W'].T[:max_count]
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    axes = axes.reshape(-1)
    for i, ax in enumerate(axes):
        ax.imshow(weights[i].reshape(28, 28), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'W{i}', fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='cnn')
    parser.add_argument('--path', default=r'.\best_models\cnn_momentum_multistep.pickle')
    args = parser.parse_args()

    os.makedirs('figs', exist_ok=True)
    if args.model == 'mlp':
        model = nn.models.Model_MLP()
    else:
        model = nn.models.Model_CNN()
    model.load_model(args.path)

    test_imgs, test_labs = load_test()
    logits = model(test_imgs)
    preds = np.argmax(logits, axis=1)
    acc = nn.metric.accuracy(logits, test_labs)
    print(f'test_acc={acc}')

    prefix = args.model
    mat = confusion_matrix(test_labs, preds)
    plot_confusion_matrix(mat, os.path.join('figs', f'confusion_matrix_{prefix}.png'))
    plot_misclassified(test_imgs, test_labs, preds, os.path.join('figs', f'misclassified_examples_{prefix}.png'))
    if args.model == 'cnn':
        plot_cnn_kernels(model, os.path.join('figs', 'cnn_kernels.png'))
    else:
        plot_mlp_weights(model, os.path.join('figs', 'mlp_weights.png'))


if __name__ == '__main__':
    main()
