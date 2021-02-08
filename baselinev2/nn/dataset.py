from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import skimage
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from average_image.bbox_utils import get_frame_annotations_and_skip_lost, get_frame_annotations
from average_image.constants import ANNOTATION_COLUMNS
from baselinev2.config import VIDEO_PATH, ANNOTATION_CSV_PATH, VIDEO_SAVE_PATH, ANNOTATION_TXT_PATH
from baselinev2.plot_utils import plot_for_video_image_and_box


def get_frames_count(video_path):
    video = cv.VideoCapture(video_path)
    count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    video.release()
    return count


def sort_annotations_by_frame_numbers(annotation_path):
    annotations = pd.read_csv(annotation_path, index_col='Unnamed: 0')
    return annotations.sort_values(by=['frame'])


def verify_annotations_processing(video_path, df, plot_scale_factor=1, desired_fps=5):
    Path(VIDEO_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))

    if w < h:
        original_dims = (
            h / 100 * plot_scale_factor, w / 100 * plot_scale_factor)
        out = cv.VideoWriter(VIDEO_SAVE_PATH + 'proof_out.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                             (h, w))

        w, h = h, w
    else:
        original_dims = (
            w / 100 * plot_scale_factor, h / 100 * plot_scale_factor)
        out = cv.VideoWriter(VIDEO_SAVE_PATH + 'proof_out.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                             (w, h))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if ret:
            annotation = get_frame_annotations_and_skip_lost(df, frame_idx)
            # annotation = get_frame_annotations(df, frame_idx)
            boxes = annotation[:, 1:5]

            fig = plot_for_video_image_and_box(gt_rgb=frame, gt_annotations=boxes, frame_number=frame_idx,
                                               original_dims=original_dims, return_figure_only=True)
            canvas = FigureCanvas(fig)
            canvas.draw()

            buf = canvas.buffer_rgba()
            out_frame = np.asarray(buf, dtype=np.uint8)[:, :, :-1]
            if out_frame.shape[0] != h or out_frame.shape[1] != w:
                out_frame = skimage.transform.resize(out_frame, (h, w))
                out_frame = (out_frame * 255).astype(np.uint8)
            out.write(out_frame)
            frame_idx += 1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()

    cv.destroyAllWindows()


if __name__ == '__main__':
    verify_annotations_processing(video_path=VIDEO_PATH, df=sort_annotations_by_frame_numbers(ANNOTATION_CSV_PATH))
    # verify_annotations_processing(video_path=VIDEO_PATH, df=pd.read_csv(ANNOTATION_CSV_PATH, index_col='Unnamed: 0'))
    # annot = pd.read_csv(ANNOTATION_TXT_PATH, sep=' ')
    # annot.columns = ANNOTATION_COLUMNS
    # verify_annotations_processing(video_path=VIDEO_PATH, df=annot)
