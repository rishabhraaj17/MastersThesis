import os

import numpy as np
import torchvision.io
import yaml
from matplotlib import pyplot as plt, patches


def plot_seg_maps(rgb, seg, fig_name, return_figure_only=False, additional_text='', cmap=None):
    if rgb.shape[-2] > rgb.shape[-3]:
        fig, ax = plt.subplots(2, 1, sharex='none', sharey='none', figsize=(12, 10))
    else:
        fig, ax = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(12, 10))

    ax_rgb, ax_seg_map = ax[0], ax[1]
    ax_rgb.imshow(rgb)
    ax_seg_map.imshow(seg, cmap=None)

    ax_rgb.set_title('RGB')
    ax_seg_map.set_title('Segmentation Map')

    fig.suptitle(f'{fig_name}\n{additional_text}')

    legends_dict = {}

    legend_patches = [patches.Patch(color=key, label=val) for key, val in legends_dict.items()]
    fig.legend(handles=legend_patches, loc=2)

    if return_figure_only:
        plt.close()
        return fig

    plt.tight_layout()
    plt.show()

    return fig


def sort_files_by_names(arr):
    files_list = [c.split('-')[0] for c in arr]
    files_list = np.array([int(c.split('/')[-1][:-4]) for c in files_list]).argsort()
    arr = np.array(arr)[files_list]
    return arr


def analyse_maps(folder_path):
    rgb_files = sort_files_by_names([os.path.join(f"{folder_path}RGB/", im) for im in os.listdir(f"{folder_path}RGB/")])
    seg_files = sort_files_by_names(
        [os.path.join(f"{folder_path}SEGMENTATION/", im) for im in os.listdir(f"{folder_path}SEGMENTATION/")])
    inst_files = sort_files_by_names(
        [os.path.join(f"{folder_path}GLAY/", im) for im in os.listdir(f"{folder_path}GLAY/")])

    for rgb_im_path, seg_im_path, inst_im_path in zip(rgb_files, seg_files, inst_files):
        rgb_im = torchvision.io.read_image(rgb_im_path).permute(1, 2, 0).numpy()
        seg_im = torchvision.io.read_image(seg_im_path).permute(1, 2, 0).numpy()
        inst_im = torchvision.io.read_image(inst_im_path).permute(1, 2, 0).numpy()

        plot_seg_maps(rgb_im, seg_im, os.path.split(rgb_im_path)[-1])

        classes = np.unique(inst_im)
        for clz in classes:
            instance_map = np.zeros_like(inst_im)
            instance_map[inst_im == clz] = 255
            plot_seg_maps(rgb_im, instance_map, os.path.split(rgb_im_path)[-1])
            print()
        print()


def dump_class_mapping(root_path, save_on_disk=False):
    data = {
        'foot_path': 0,
        'street': 10,
        'grass_path': 20,
        'parking': 30,
        'lane_switch': 40,
        'tree': 50,
        'building': 60
    }
    if save_on_disk:
        with open(f'{root_path}class_mappings.yaml', 'w+') as f:
            yaml.dump(data, f)
    return data


def dump_image_mapping(root_path, save_on_disk=False):
    data = {
        'bookstore': {
            0: ('test', '0.png'),
            1: ('train', '00.png'),
            2: ('train', '01.png'),
            3: ('train', '02.png'),
            4: ('train', '03.png'),
            5: ('train', '04.png'),
            6: ('train', '05.png')
        },
        'coupa': {
            0: ('test', '1.png'),
            1: ('train', '06.png'),
            2: ('train', '07.png'),
            3: ('train', '08.png')
        },
        'deathCircle': {
            0: ('test', '2.png'),
            1: ('train', '09.png'),
            2: ('train', '10.png'),
            3: ('train', '11.png'),
            4: ('train', '12.png'),
        },
        'gates': {
            0: ('test', '3.png'),
            1: ('train', '13.png'),
            2: ('train', '14.png'),
            3: ('train', '15.png'),
            4: ('train', '16.png'),
            5: ('train', '17.png'),
            6: ('train', '18.png'),
            7: ('train', '19.png'),
            8: ('train', '20.png')
        },
        'hyang': {
            0: ('test', '4.png'),
            1: ('train', '21.png'),
            2: ('train', '22.png'),
            3: ('train', '23.png'),
            4: ('train', '24.png'),
            5: ('train', '25.png'),
            6: ('train', '26.png'),
            7: ('train', '27.png'),
            8: ('train', '28.png'),
            9: ('train', '29.png'),
            10: ('train', '30.png'),
            11: ('train', '31.png'),
            12: ('train', '32.png'),
            13: ('train', '33.png'),
            14: ('train', '34.png')
        },
        'little': {
            0: ('test', '5.png'),
            1: ('train', '35.png'),
            2: ('train', '36.png'),
            3: ('train', '37.png'),
        },
        'nexus': {
            0: ('test', '6.png'),
            1: ('train', '38.png'),
            2: ('train', '39.png'),
            3: ('train', '40.png'),
            4: ('train', '41.png'),
            5: ('train', '42.png'),
            6: ('train', '43.png'),
            7: ('train', '44.png'),
            8: ('train', '45.png'),
            9: ('train', '46.png'),
            10: ('train', '47.png'),
            11: ('train', '48.png'),
        },
        'quad': {
            0: ('test', '7.png'),
            1: ('train', '49.png'),
            2: ('train', '50.png'),
            3: ('train', '51.png'),
        }
    }
    if save_on_disk:
        with open(f'{root_path}img_mappings.yaml', 'w+') as f:
            yaml.dump(data, f)
    return data


if __name__ == '__main__':
    root = '../../Datasets/SDD_SEG_MAPS/'
    train_folder = f'{root}train/'
    test_folder = f'{root}test/'

    # analyse_maps(test_folder)
    # dump_class_mapping(root)
    dump_image_mapping(root)
