import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-option1', action='store_const', const=True, default=False, help='test accuracy - training loss')
    parser.add_argument('-option2', action='store_const', const=True, default=False, help='test accuracy - test loss')
    parser.add_argument('-option3', action='store_const', const=True, default=False, help='test loss - train loss')
    parser.add_argument('-n_cols', default=4, type=int, help='n_cols in legend')
    parser.add_argument('-grid_off', action='store_const', const=True, default=False, help='no grid in plots')
    parser.add_argument('-cv_res', action='store_const', const=True, default=False, help='if using results from crossval')

    return parser.parse_known_args()


def get_accuracy_results(path):
    results = []
    splits = 0
    mean_accuracy = []
    with open(path) as reader:
        for line in reader:
            if 'Average acc.' in line:
                line = line.split(sep=',')[0].split(sep=':')[1]
                result = line.replace(' ', '')
                results.append(float(result))

            elif 'split {0} done'.format(splits) in line:
                splits += 1
            elif 'Mean Accuracy' in line:
                mean_accuracy.append(float(line.split(': ')[1].replace(' ', '')))

    return np.split(np.asarray(results, dtype=float), max(1, splits)), np.asarray(mean_accuracy, dtype=float)


def get_test_losses(path):
    results = []
    splits = 0
    with open(path) as reader:
        for line in reader:
            if 'Average acc.' in line:
                line = line.split(sep=',')[1].split(sep=':')[1]
                result = line.replace(' ', '').replace('\n', '')
                results.append(float(result))

            elif 'split {0} done'.format(splits) in line:
                splits += 1

    return np.split(np.asarray(results, dtype=float), max(1, splits))


def get_training_losses(path):
    results = []
    splits = 0
    with open(path) as reader:
        for line in reader:
            if 'Training Loss' in line:
                line = line.split(sep=':')[1]
                result = line.replace(' ', '')
                results.append(float(result))

            elif 'split {0} done'.format(splits) in line:
                splits += 1

    return np.split(np.asarray(results, dtype=float), max(1, splits))


def get_color(normalize=0, boost_channel=0):
    rgb = np.random.randint(0, 255, 3, dtype=int)
    rgb[boost_channel] = 150  # boost channel for differentiation
    if normalize:
        rgb = rgb / 255.0
    return rgb


def main(path):
    options, _ = parse_args()
    files = glob.glob(path+'/*')

    n = len(files)
    # define the colormap
    if n >= 10:  # get new colors if more than 10 files
        cmap = plt.get_cmap('rainbow')
        colors = cmap(np.linspace(0, 1, n-10, endpoint=False))
        color_iter = iter(colors)
    elif options.cv_res and n >= 4:
        cmap = plt.get_cmap('rainbow')
        colors = cmap(np.linspace(0, 1, 3*n - 10, endpoint=False))
        color_iter = iter(colors)

    n = 1
    if options.option1 or options.option2 or options.option3:
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(12, 6.5), dpi=100)
        # fig.suptitle(f'ZS Results - AwA2 Dataset - {path[:3]} - lr: {path[4:8]} - surj: {path[-5:-1]}', fontsize=14)  # specific
        fig.suptitle(f'ZS Results - AwA2 Dataset - {path}', fontsize=14)  # any

        for i, file in enumerate(sorted(files)):
            if options.option1:
                accuracy, mean_accuracy = get_accuracy_results(file)
                train_loss = get_training_losses(file)

                ax1.set_ylim([0, 0.8])
                ax2.set_ylim([0, 1])
                for j in range(len(accuracy)):
                    label = file.split('/')[2]
                    train_loss[j] /= np.amax(train_loss[j])
                    if len(mean_accuracy) > 0:
                        label += ' - (' + '{:.3f}'.format(mean_accuracy[j]) + ')'
                    if n > 10:
                        c = next(color_iter)
                        ax1.plot(range(5, 5 * len(accuracy[j]) + 1, 5), accuracy[j], '-', label=label, color=c)
                        ax2.plot(range(1, len(train_loss[j]) + 1), train_loss[j], '-', label=label, color=c)
                    else:
                        ax1.plot(range(5, 5 * len(accuracy[j]) + 1, 5), accuracy[j], '-', label=label)
                        ax2.plot(range(1, len(train_loss[j]) + 1), train_loss[j], '-', label=label)

                    n += 1

                ax1.set_ylabel('Test Accuracy')
                ax2.set_ylabel('Training Loss')

            elif options.option2:
                accuracy, mean_accuracy = get_accuracy_results(file)
                test_losses = get_test_losses(file)

                ax1.set_ylim([0, 0.8])
                ax2.set_ylim([0, 1])
                for j in range(len(accuracy)):
                    label = file.split('/')[2]
                    test_losses[j] /= np.amax(test_losses[j])
                    if len(mean_accuracy) > 0:
                        label += ' - (' + '{:.3f}'.format(mean_accuracy[j]) + ')'
                    if n > 10:
                        c = next(color_iter)
                        ax1.plot(range(5, 5 * len(accuracy) + 1, 5), accuracy, '-', label=label, color=c)
                        ax2.plot(range(5, 5 * len(test_losses[j]) + 1, 5), test_losses[j], '-', label=file.split('/')[2], color=c)
                    else:
                        ax1.plot(range(5, 5 * len(accuracy) + 1, 5), accuracy, '-', label=label)
                        ax2.plot(range(5, 5 * len(test_losses[j]) + 1, 5), test_losses[j], '-', label=file.split('/')[2])

                    n += 1

                ax1.set_ylabel('Test Accuracy')
                ax2.set_ylabel('Test Loss')
            else:
                test_losses = get_test_losses(file)
                train_loss = get_training_losses(file)

                ax1.set_ylim([0, 1])
                ax2.set_ylim([0, 1])
                for j in range(len(test_losses)):
                    test_losses[j] /= np.amax(test_losses[j])
                    train_loss[j] /= np.amax(train_loss[j])
                    if n > 10:
                        c = next(color_iter)
                        ax1.plot(range(5, 5 * len(test_losses[j]) + 1, 5), test_losses[j], '-', label=file.split('/')[2], color=c)
                        ax2.plot(range(1, len(train_loss[j]) + 1), train_loss[j], '-', label=file.split('/')[2], color=c)
                    else:
                        ax1.plot(range(5, 5 * len(test_losses[j]) + 1, 5), test_losses[j], '-', label=file.split('/')[2])
                        ax2.plot(range(1, len(train_loss[j]) + 1), train_loss[j], '-', label=file.split('/')[2])

                    n += 1

                ax1.set_ylabel('Test Loss')
                ax2.set_ylabel('Training Loss')

        if not options.grid_off:
            ax1.grid(which='both')
            ax2.grid(which='both')

    else:
        fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True, figsize=(12, 6.5), dpi=100)
        # fig.suptitle(f'ZS Results - AwA2 Dataset - {path[:3]} - lr: {path[4:8]} - surj: {path[-5:-1]}', fontsize=14)  # specific
        fig.suptitle(f'ZS Results - AwA2 Dataset - {path}', fontsize=14)  # any

        for i, file in enumerate(sorted(files)):
            accuracy, mean_accuracy = get_accuracy_results(file)
            test_losses = get_test_losses(file)
            train_loss = get_training_losses(file)

            ax1.set_ylim([0, 0.8])
            ax2.set_ylim([0, 1])
            ax3.set_ylim([0, 1])
            for j in range(len(accuracy)):
                label = file.split('/')[2]
                test_losses[j] /= np.amax(test_losses[j])
                train_loss[j] /= np.amax(train_loss[j])
                if len(mean_accuracy) > 0:
                    label += ' - (' + '{:.3f}'.format(mean_accuracy[j]) + ')'
                if n > 10:
                    c = next(color_iter)
                    ax1.plot(range(5, 5*len(accuracy[j])+1, 5), accuracy[j], '-', label=label, color=c)
                    ax2.plot(range(5, 5*len(test_losses[j])+1, 5), test_losses[j], '-', label=file.split('/')[2], color=c)
                    ax3.plot(range(1, len(train_loss[j])+1), train_loss[j], '-', label=file.split('/')[2], color=c)
                else:
                    ax1.plot(range(5, 5*len(accuracy[j])+1, 5), accuracy[j], '-', label=label)
                    ax2.plot(range(5, 5*len(test_losses[j])+1, 5), test_losses[j], '-', label=file.split('/')[2])
                    ax3.plot(range(1, len(train_loss[j])+1), train_loss[j], '-', label=file.split('/')[2])

                n += 1

        ax1.set_ylabel('Test Accuracy')
        ax2.set_ylabel('Test Loss')
        ax3.set_ylabel('Training Loss')

        if not options.grid_off:
            ax1.grid(which='both')
            ax2.grid(which='both')
            ax3.grid(which='both')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=options.n_cols, prop={'size': 8}, frameon=False)
    plt.xlabel('Epochs')
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(bottom=0.2)

    plt.show()


if __name__ == '__main__':
    path = sys.argv[1]
    main(path)

