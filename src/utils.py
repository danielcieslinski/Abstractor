import numpy as np
import pandas as pd
from os import listdir
from os import path
import json
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle

def load_folder(folder):
    x = []

    files = sorted(listdir(folder))
    for f in files:
        with open(folder+f)as fl:
            x.append(json.load(fl))

    return x

def plot_color_range():
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    # 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
    # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
    plt.figure(figsize=(5, 2), dpi=200)
    plt.imshow([list(range(10))], cmap=cmap, norm=norm)
    plt.xticks(list(range(10)))
    plt.yticks([])
    plt.show()

def plot_task(task):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1

    plt.tight_layout()
    plt.show()

def get_data(relpath='../res/'):
    """
    :param relpath: relative path to resource folder
    :return: train, test, eval lists
    """

    if path.exists(relpath + 'data.pickle'):
        with open(relpath+'data.pickle','rb') as f:
            return pickle.load(f)


    data = [] #Must be: [tr, te, ev]
    for t in ['training/', 'evaluation/', 'test/']:
        p = relpath+t
        data.append(load_folder(p))

    with open(relpath+'data.pickle', 'wb') as f:
        pickle.dump(data, f)

    return data

def explore_data(tr, te, ev):

    all_shapes = []

    for i, t in enumerate(tr):
        train_size = np.shape(t['train'])
        test_size = np.shape(t['test'])
        print(i, 'train', train_size, 'test', test_size)
        input_shapes, output_shapes = [], []
        for e in t['train']:
            input_shapes.append(np.shape(e['input']))
            output_shapes.append(np.shape(e['output']))

        # train only
        all_shapes.append([input_shapes, output_shapes])

        print('    ', 'input shapes', input_shapes)
        print('    ', 'output shapes', output_shapes)

    print(max(all_shapes))
    print(min(all_shapes))

def main():
    tr, te, ev = get_data()
    explore_data(tr, te, ev)



if __name__ == '__main__':
    main()


