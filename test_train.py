import argparse
import os
from struct import unpack
import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot

np.random.seed(309)


def load_train_valid():
    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    idx = np.random.permutation(np.arange(num))
    with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    train_imgs = train_imgs / train_imgs.max()
    valid_imgs = valid_imgs / valid_imgs.max()
    return [train_imgs, train_labs], [valid_imgs, valid_labs]


def build_model(model_type):
    if model_type == 'mlp':
        init = lambda size: np.random.normal(size=size) * 0.1
        model = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
        for layer in model.layers:
            if layer.optimizable:
                layer.W = init(layer.W.shape)
                layer.b = init(layer.b.shape)
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
        return model
    if model_type == 'cnn':
        return nn.models.Model_CNN()
    raise ValueError(f'Unknown model type: {model_type}')


def build_optimizer(optimizer_type, model, lr, mu):
    if optimizer_type == 'sgd':
        return nn.optimizer.SGD(init_lr=lr, model=model)
    if optimizer_type == 'momentum':
        return nn.optimizer.MomentGD(init_lr=lr, model=model, mu=mu)
    raise ValueError(f'Unknown optimizer type: {optimizer_type}')


def build_scheduler(scheduler_type, optimizer):
    if scheduler_type == 'none':
        return None
    if scheduler_type == 'multistep':
        return nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
    if scheduler_type == 'exponential':
        return nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9997)
    raise ValueError(f'Unknown scheduler type: {scheduler_type}')


def save_curve(runner, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.set_tight_layout(True)
    plot(runner, axes)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--optimizer', choices=['sgd', 'momentum'], default='sgd')
    parser.add_argument('--scheduler', choices=['none', 'multistep', 'exponential'], default='multistep')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.06)
    parser.add_argument('--mu', type=float, default=0.9)
    parser.add_argument('--log-iters', type=int, default=100)
    args = parser.parse_args()

    os.makedirs('best_models', exist_ok=True)
    os.makedirs('figs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    train_set, valid_set = load_train_valid()
    model = build_model(args.model)
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.mu)
    scheduler = build_scheduler(args.scheduler, optimizer)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=args.batch_size, scheduler=scheduler)

    run_name = f'{args.model}_{args.optimizer}_{args.scheduler}'
    save_dir = os.path.join('best_models', run_name)
    runner.train(train_set, valid_set, num_epochs=args.epochs, log_iters=args.log_iters, save_dir=save_dir)

    model_path = os.path.join('best_models', f'{run_name}.pickle')
    model.save_model(model_path)
    curve_path = os.path.join('figs', f'{run_name}_curve.png')
    save_curve(runner, curve_path)

    result_path = os.path.join('results', f'{run_name}.txt')
    with open(result_path, 'w') as f:
        f.write(f'run_name={run_name}\n')
        f.write(f'best_valid_acc={runner.best_score}\n')
        f.write(f'final_lr={optimizer.init_lr}\n')
        f.write(f'curve={curve_path}\n')
        f.write(f'model={model_path}\n')
    print(f'Saved model to {model_path}')
    print(f'Saved curve to {curve_path}')
    print(f'Saved result to {result_path}')


if __name__ == '__main__':
    main()
