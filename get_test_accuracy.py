import numpy as np
import matplotlib.pyplot as plt
import sys
import glob


def get_training_losses(path):
    results = []
    test_accuracy = 0
    with open(path) as reader:
        for line in reader:
            if 'Training Loss' in line:
                line = line.split(sep=':')[1]
                result = line.replace(' ', '')
                results.append(float(result))
            elif 'Accuracy:' in line:
                line = line.split(sep=':')[1]
                test_accuracy = float(line.replace(' ', ''))

    return np.asarray(results, dtype=float), test_accuracy


def main():
    path = sys.argv[1]
    files = glob.glob(path+'/*')

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 6.5), dpi=100)
    fig.suptitle(f'ZS Results - AwA2 Dataset - {path}', fontsize=14)  # any

    for i, file in enumerate(sorted(files)):
        train_loss, test_accuracy = get_training_losses(file)

        ax.set_ylim([0, 1])

        label = file.split('/')[2]
        label += ' - (' + '{:.3f}'.format(test_accuracy) + ')'
        train_loss /= np.amax(train_loss)
        ax.plot(range(1, len(train_loss) + 1), train_loss, '-', label=label)

    ax.set_ylabel('Training Loss')
    ax.grid(which='both')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=int(sys.argv[2]), prop={'size': 8}, frameon=False)
    plt.xlabel('Epochs')
    plt.subplots_adjust(bottom=0.2)

    plt.show()


if __name__ == '__main__':
    main()
