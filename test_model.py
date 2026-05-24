import argparse
from struct import unpack
import gzip

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--path', default=r'.\best_models\mlp_sgd_multistep.pickle')
    args = parser.parse_args()

    if args.model == 'mlp':
        model = nn.models.Model_MLP()
    else:
        model = nn.models.Model_CNN()
    model.load_model(args.path)

    test_imgs, test_labs = load_test()
    logits = model(test_imgs)
    acc = nn.metric.accuracy(logits, test_labs)
    print(f'test_acc={acc}')


if __name__ == '__main__':
    main()
