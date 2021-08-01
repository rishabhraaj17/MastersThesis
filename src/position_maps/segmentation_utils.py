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


def dump_class_mapping(root_path):
    data = {
        'foot_path': 0,
        'street': 10,
        'grass_path': 20,
        'parking': 30,
        'lane_switch': 40,
        'tree': 50,
        'building': 60
    }
    with open(f'{root_path}class_mappings.yaml', 'w+') as f:
        yaml.dump(data, f)
    return data


def dump_image_mapping(root_path):
    data = {
        'bookstore:': {
            0: ('test', '0.jpg'),
            1: ('train', '00.jpg'),
            2: ('train', '01.jpg'),
            3: ('train', '02.jpg'),
            4: ('train', '03.jpg'),
            5: ('train', '04.jpg'),
            6: ('train', '05.jpg')
        },
        'coupa': {
            0: ('test', '1.jpg'),
            1: ('train', '06.jpg'),
            2: ('train', '07.jpg'),
            3: ('train', '08.jpg')
        },
        'deathCircle': {
            0: ('test', '2.jpg'),
            1: ('train', '09.jpg'),
            2: ('train', '10.jpg'),
            3: ('train', '11.jpg'),
            4: ('train', '12.jpg'),
        },
        'gates': {
            0: ('test', '3.jpg'),
            1: ('train', '13.jpg'),
            2: ('train', '14.jpg'),
            3: ('train', '15.jpg'),
            4: ('train', '16.jpg'),
            5: ('train', '17.jpg'),
            6: ('train', '18.jpg'),
            7: ('train', '19.jpg'),
            8: ('train', '20.jpg')
        },
        'hyang': {
            0: ('test', '4.jpg'),
            1: ('train', '21.jpg'),
            2: ('train', '22.jpg'),
            3: ('train', '23.jpg'),
            4: ('train', '24.jpg'),
            5: ('train', '25.jpg'),
            6: ('train', '26.jpg'),
            7: ('train', '27.jpg'),
            8: ('train', '28.jpg'),
            9: ('train', '29.jpg'),
            10: ('train', '30.jpg'),
            11: ('train', '31.jpg'),
            12: ('train', '32.jpg'),
            13: ('train', '33.jpg'),
            14: ('train', '34.jpg')
        },
        'little': {
            0: ('test', '5.jpg'),
            1: ('train', '35.jpg'),
            2: ('train', '36.jpg'),
            3: ('train', '37.jpg'),
        },
        'nexus': {
            0: ('test', '6.jpg'),
            1: ('train', '38.jpg'),
            2: ('train', '39.jpg'),
            3: ('train', '40.jpg'),
            4: ('train', '41.jpg'),
            5: ('train', '42.jpg'),
            6: ('train', '43.jpg'),
            7: ('train', '44.jpg'),
            8: ('train', '45.jpg'),
            9: ('train', '46.jpg'),
            10: ('train', '47.jpg'),
            11: ('train', '48.jpg'),
        },
        'quad': {
            0: ('test', '7.jpg'),
            1: ('train', '49.jpg'),
            2: ('train', '50.jpg'),
            3: ('train', '51.jpg'),
        }
    }
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
