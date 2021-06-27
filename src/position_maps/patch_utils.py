import matplotlib.pyplot as plt
import torch
import torchvision.io
from mmseg.models import VisionTransformer
from torch import nn
from torch.nn.functional import interpolate

from src_lib.models_hub.trans_unet import get_r50_b16_config, VisionTransformer as ViT
from src.position_maps.utils import generate_position_map


def extract_patches_2d(img, patch_shape, step=[1.0, 1.0], batch_first=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if (img.size(2) < patch_H):
        num_padded_H_Top = (patch_H - img.size(2)) // 2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)
    if (img.size(3) < patch_W):
        num_padded_W_Left = (patch_W - img.size(3)) // 2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left, num_padded_W_Right, 0, 0), 0)
        img = padding_W(img)
    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0]) if (isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W * step[1]) if (isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if ((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H, img[:, :, -patch_H:, ].permute(0, 1, 3, 2).unsqueeze(2)), dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if ((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat(
            (patches_fold_HW, patches_fold_H[:, :, :, -patch_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3)), dim=3)
    patches = patches_fold_HW.permute(2, 3, 0, 1, 4, 5)
    patches = patches.reshape(-1, img.size(0), img.size(1), patch_H, patch_W)
    if (batch_first):
        patches = patches.permute(1, 0, 2, 3, 4)
    return patches


def reconstruct_from_patches_2d(patches, img_shape, step=[1.0, 1.0], batch_first=False):
    if (batch_first):
        patches = patches.permute(1, 0, 2, 3, 4)
    patch_H, patch_W = patches.size(3), patches.size(4)
    img_size = (patches.size(1), patches.size(2), max(img_shape[0], patch_H), max(img_shape[1], patch_W))
    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0]) if (isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W * step[1]) if (isinstance(step[1], float)) else step[1]
    nrow, ncol = 1 + (img_size[-2] - patch_H) // step_int[0], 1 + (img_size[-1] - patch_W) // step_int[1]
    r_nrow = nrow + 1 if ((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    r_ncol = ncol + 1 if ((img_size[3] - patch_W) % step_int[1] != 0) else ncol
    patches = patches.reshape(r_nrow, r_ncol, img_size[0], img_size[1], patch_H, patch_W)
    img = torch.zeros(img_size, device=patches.device)
    overlap_counter = torch.zeros(img_size, device=patches.device)
    for i in range(nrow):
        for j in range(ncol):
            img[:, :, i * step_int[0]:i * step_int[0] + patch_H, j * step_int[1]:j * step_int[1] + patch_W] += patches[
                i, j,]
            overlap_counter[:, :, i * step_int[0]:i * step_int[0] + patch_H,
            j * step_int[1]:j * step_int[1] + patch_W] += 1
    if ((img_size[2] - patch_H) % step_int[0] != 0):
        for j in range(ncol):
            img[:, :, -patch_H:, j * step_int[1]:j * step_int[1] + patch_W] += patches[-1, j,]
            overlap_counter[:, :, -patch_H:, j * step_int[1]:j * step_int[1] + patch_W] += 1
    if ((img_size[3] - patch_W) % step_int[1] != 0):
        for i in range(nrow):
            img[:, :, i * step_int[0]:i * step_int[0] + patch_H, -patch_W:] += patches[i, -1,]
            overlap_counter[:, :, i * step_int[0]:i * step_int[0] + patch_H, -patch_W:] += 1
    if ((img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0):
        img[:, :, -patch_H:, -patch_W:] += patches[-1, -1,]
        overlap_counter[:, :, -patch_H:, -patch_W:] += 1
    img /= overlap_counter
    if (img_shape[0] < patch_H):
        num_padded_H_Top = (patch_H - img_shape[0]) // 2
        num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
        img = img[:, :, num_padded_H_Top:-num_padded_H_Bottom, ]
    if (img_shape[1] < patch_W):
        num_padded_W_Left = (patch_W - img_shape[1]) // 2
        num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
        img = img[:, :, :, num_padded_W_Left:-num_padded_W_Right]
    return img


def quick_viz(im1, im2=None, img_idx=0):
    im1 = preprocess_image(im1, img_idx)

    if im2 is None:
        fig, ax = plt.subplots(1, 1, sharex='none', sharey='none', figsize=(14, 12))
        image_axis = ax
    else:
        im2 = preprocess_image(im2, img_idx)

        fig, ax = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(14, 12))
        image_axis, stitched_image_axis = ax

        stitched_image_axis.imshow(im2)
        stitched_image_axis.set_title('Stitched')

    image_axis.imshow(im1)
    image_axis.set_title('Original')

    plt.tight_layout()
    plt.show()


def preprocess_image(im, img_idx):
    im = im.detach().cpu() if isinstance(im, torch.Tensor) else im
    im = im[img_idx] if im.ndim == 4 else im
    im = im.permute(1, 2, 0) if im.shape[0] in [1, 2, 3] else im
    return im


if __name__ == '__main__':
    im_path = "/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD/annotations/" \
              "deathCircle/video1/reference.jpg"
    random_idx = 0
    patch_size = (256, 256)
    batch_size = 2
    num_locs = 40

    img = torchvision.io.read_image(im_path).unsqueeze(0)
    img = interpolate(img, scale_factor=0.5).repeat(batch_size, 1, 1, 1)

    patches = extract_patches_2d(img, patch_size, batch_first=True)
    stitched_img = reconstruct_from_patches_2d(patches, img.shape[-2:], batch_first=True).to(dtype=torch.uint8)

    p = torchvision.utils.make_grid(patches[random_idx].squeeze(0), nrow=img.shape[2] // patch_size[0])

    # quick_viz(p)
    # quick_viz(img, stitched_img)

    loc_x = torch.randint(0, img.shape[-2], (num_locs,))
    loc_y = torch.randint(0, img.shape[-1], (num_locs,))
    locs = torch.stack((loc_x, loc_y)).t()
    p_img = torch.from_numpy(
        generate_position_map(
            list(img.shape[-2:]), locs, sigma=2, return_combined=True))[(None,) * 2].repeat(batch_size, 1, 1, 1)

    patches_target = extract_patches_2d(p_img, patch_size, batch_first=True)
    stitched_img_target = reconstruct_from_patches_2d(patches_target, p_img.shape[-2:], batch_first=True)

    p_target = torchvision.utils.make_grid(patches_target[random_idx].squeeze(0), nrow=p_img.shape[2] // patch_size[0])

    # quick_viz(p_target)
    # quick_viz(p_img, stitched_img_target)

    conf = get_r50_b16_config()
    conf.n_classes = 1
    # # from mm-seg -> reshape to input shape -> put Seg head to get desired classes
    # m = VisionTransformer(img_size=patch_size)

    m = ViT(conf, img_size=patch_size, num_classes=1, use_attn_decoder=True)

    inp = (patches.contiguous().view(-1, *patches.shape[2:]).float() / 255.0)
    o = m(inp)
    o = o.view(batch_size, -1, 1, *patches.shape[3:])

    stitched_output = reconstruct_from_patches_2d(o, img.shape[-2:], batch_first=True)
    quick_viz(img, stitched_output)

    print()
