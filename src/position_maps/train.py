import albumentations as A
import hydra
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from average_image.constants import SDDVideoClasses
from position_maps.dataset import SDDFrameAndAnnotationDataset
from position_maps.utils import resize_transform

seed_everything(42)


@hydra.main(config_path="config", config_name="config")
def train(cfg):
    transform = A.Compose(
        [A.Resize(height=400, width=400)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
        keypoint_params=A.KeypointParams(format='xy')
    )

    train_dataset = SDDFrameAndAnnotationDataset(
        root=cfg.root, video_label=getattr(SDDVideoClasses, cfg.video_class),
        num_videos=cfg.train.num_videos, transform=transform,
        num_workers=cfg.dataset_workers, scale=cfg.scale_factor,
        video_number_to_use=cfg.train.video_number_to_use,
        multiple_videos=cfg.train.multiple_videos,
        use_generated=cfg.use_generated_dataset,
        sigma=cfg.sigma,
        plot=cfg.plot_samples,
        desired_size=cfg.desired_size
    )
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for d in loader:
        print()


if __name__ == '__main__':
    train()
