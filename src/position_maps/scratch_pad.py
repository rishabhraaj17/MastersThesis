import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops
from kornia.losses import BinaryFocalLossWithLogits
# from mmseg.models import VisionTransformer, HRNet
from mmdet.models.utils.gaussian_target import get_local_maximum, get_topk_from_heatmap
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Subset
# from torchvision.models.detection.rpn import RegionProposalNetwork
# from mmdet.models.backbones.hourglass import HourglassNet
# from mmdet.models.dense_heads.centernet_head import CenterNetHead
# from mmdet.models.losses.gaussian_focal_loss import GaussianFocalLoss
# from mmcv.ops.deform_conv import DeformConv2d
# from mmdet.models.utils import gen_gaussian_target
# from mmdet.models.utils.gaussian_target import get_topk_from_heatmap, get_local_maximum
# from mmpose.datasets.pipelines import HeatmapGenerator, TopDownGenerateTarget
from tqdm import tqdm

from baselinev2.plot_utils import plot_one_with_bounding_boxes
from evaluate import setup_multiple_test_datasets
from losses import CenterNetFocalLoss
from utils import heat_map_collate_fn, ImagePadder, plot_predictions_v2
from patch_utils import quick_viz
from src_lib.models_hub import DeepLabV3


@hydra.main(config_path="config", config_name="config")
def patch_experiment(cfg):
    device = 'cuda:0'
    test_dataset, target_max_shape = setup_multiple_test_datasets(cfg, return_dummy_transform=False)

    indices = np.random.choice(len(test_dataset), len(test_dataset) // 4, replace=False)
    test_dataset = Subset(test_dataset, indices=indices)

    test_loader = DataLoader(test_dataset, batch_size=cfg.eval.batch_size, shuffle=False,
                             num_workers=cfg.eval.num_workers, collate_fn=heat_map_collate_fn,
                             pin_memory=cfg.eval.pin_memory, drop_last=cfg.eval.drop_last)

    loss_fn = BinaryFocalLossWithLogits(
        alpha=cfg.eval.loss.bfl.alpha, gamma=cfg.eval.loss.bfl.gamma, reduction=cfg.eval.loss.reduction)
    gauss_loss_fn = [CenterNetFocalLoss()]

    model = DeepLabV3(
        config=cfg, train_dataset=None, val_dataset=None,
        loss_function=loss_fn, collate_fn=heat_map_collate_fn, additional_loss_functions=gauss_loss_fn,
        desired_output_shape=None)

    load_dict = torch.load('death_circle4_deeplabv3.pt', map_location=device)
    model.load_state_dict(load_dict)

    model.to(device)
    model.eval()

    for idx, data in enumerate(tqdm(test_loader)):
        frames, heat_masks, position_map, distribution_map, class_maps, meta = data

        padder = ImagePadder(frames.shape[-2:], factor=cfg.eval.preproccesing.pad_factor)
        frames, heat_masks = padder.pad(frames)[0], padder.pad(heat_masks)[0]
        frames, heat_masks = frames.to(device), heat_masks.to(device)

        with torch.no_grad():
            out = model(frames)

        loss1 = getattr(torch.Tensor, cfg.eval.loss.reduction)(model.calculate_loss(out, heat_masks))
        loss2 = getattr(torch.Tensor, cfg.eval.loss.reduction)(model.calculate_additional_losses(
            out, heat_masks, cfg.eval.loss.gaussian_weight, cfg.eval.loss.apply_sigmoid))
        loss = loss1 + loss2
        # print(loss.item())

        out = [o.cpu().squeeze(1) for o in out]

        random_idx = np.random.choice(frames.shape[0], 1, replace=False).item()
        plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                            heat_masks[random_idx].squeeze().cpu(),
                            torch.nn.functional.threshold(out[0][random_idx].sigmoid(),
                                                          threshold=cfg.prediction.threshold,
                                                          value=cfg.prediction.fill_value,
                                                          inplace=True),
                            logits_mask=out[0][random_idx].sigmoid(),
                            additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                            f"| Epoch: {idx} "
                                            f"| Threshold: {cfg.prediction.threshold}")
        plot_predictions_v2(frames[random_idx].squeeze().cpu().permute(1, 2, 0),
                            heat_masks[random_idx].squeeze().cpu(),
                            torch.nn.functional.threshold(out[-1][random_idx].sigmoid(),
                                                          threshold=cfg.prediction.threshold,
                                                          value=cfg.prediction.fill_value,
                                                          inplace=True),
                            logits_mask=out[-1][random_idx].sigmoid(),
                            additional_text=f"{model._get_name()} | {loss_fn._get_name()} "
                                            f"| Epoch: {idx} "
                                            f"| Threshold: {cfg.prediction.threshold}")

        # get locations from heatmap
        kernel = 3
        loc_cutoff = 0.05
        marker_size = 3
        # padder = ImagePadder(out[-1].shape[-2:], factor=3)
        # out = [padder.pad(o.unsqueeze(1))[0].sigmoid().squeeze(1) for o in out]

        out = [o.sigmoid() for o in out]

        pruned_locations = []
        loc_maxima_per_output = [get_local_maximum(o, kernel) for o in out]
        for li, loc_max_out in enumerate(loc_maxima_per_output):
            temp_locations = []
            for out_img_idx in range(loc_max_out.shape[0]):
                h_loc, w_loc = torch.where(loc_max_out[out_img_idx] > loc_cutoff)
                loc = torch.stack((w_loc, h_loc)).t()

                temp_locations.append(loc)

                # viz
                plt.imshow(frames[out_img_idx].cpu().permute(1, 2, 0))
                plt.plot(w_loc, h_loc, 'o', markerfacecolor='r', markeredgecolor='k', markersize=marker_size)

                plt.title(f'Out - {li} - {out_img_idx}')
                plt.tight_layout()
                plt.show()

            pruned_locations.append(temp_locations)

        crop_h, crop_w = 128, 128
        for l_idx, (loc_from_0, loc_from_1) in enumerate(zip(*pruned_locations)):
            locations = loc_from_0 if loc_from_0.shape[0] > loc_from_1.shape[0] else loc_from_1
            crop_box_cxcywh = torch.stack([torch.tensor([kp[0], kp[1], crop_w, crop_h]) for kp in locations])
            crop_box_ijwh = torchvision.ops.box_convert(crop_box_cxcywh, 'cxcywh', 'xywh')

            crop_box_ijwh = torch.stack([torch.tensor([b[1], b[0], b[2], b[3]]) for b in crop_box_ijwh])

            crops = [torchvision.transforms.F.crop(
                frames[l_idx].unsqueeze(0).cpu(), box[0].item(), box[1].item(), box[2].item(), box[3].item())
                for box in crop_box_ijwh.to(dtype=torch.int32)]
            target_crops = [torchvision.transforms.F.crop(
                heat_masks[l_idx].unsqueeze(0).cpu(), box[0].item(), box[1].item(), box[2].item(), box[3].item())
                for box in crop_box_ijwh.to(dtype=torch.int32)]

            crops_filtered, target_crops_filtered = [], []
            for c, tc in zip(crops, target_crops):
                if c.numel() != 0:
                    if not (c.shape[-1] == crop_w) or not (c.shape[-2] == crop_h):
                        diff_h = crop_h - c.shape[2]
                        diff_w = crop_w - c.shape[3]

                        c = pad(c, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
                                mode='replicate')
                        tc = pad(tc, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
                                 mode='constant')
                    crops_filtered.append(c)
                    target_crops_filtered.append(tc)

            crops_filtered = torch.cat(crops_filtered)
            target_crops = torch.cat(target_crops_filtered)

            grid = torchvision.utils.make_grid(crops_filtered)

            plt.imshow(grid.permute(1, 2, 0))
            plt.show()

            target_grid = torchvision.utils.make_grid(target_crops)

            plt.imshow(target_grid.permute(1, 2, 0))
            plt.show()

            plot_one_with_bounding_boxes(frames[l_idx].permute(1, 2, 0).cpu(), crop_box_ijwh)
            print()

        print()
    print()


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # plt.imshow(torch.randn((25, 25)))
    # plt.show()
    patch_experiment()
    # a = torch.randn((1, 3, 720, 360))
    # h_net = HourglassNet()
    # center_net_head = CenterNetHead(in_channel=256, feat_channel=64, num_classes=1)
    # o = h_net(a)
    # o = center_net_head(o)
    # hg = HeatmapGenerator(output_size=720, num_joints=4, sigma=2)

    # # make heatmaps
    # locs = [[30, 40], [300, 400], [128, 230], [500, 500], [10, 500]]
    # maps = torch.stack([gen_gaussian_target(torch.zeros((720, 640)), l, radius=1) for l in locs])
    # m = maps.sum(0).numpy()
    # get locations from heatmap
    # loc_max = get_local_maximum(maps, 3)
    # lm = loc_max.sum(0).numpy()
    # print(np.where(lm))
    # # get top-k locations from heatmap
    # print('top-k:')
    # top_k_scores, top_k_idx, top_k_clses, top_k_ys, top_k_xs = get_topk_from_heatmap(maps.unsqueeze(0), k=100)
    # print(top_k_xs, top_k_ys)
    # print()