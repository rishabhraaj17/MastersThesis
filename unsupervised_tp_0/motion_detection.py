from copy import copy
import cv2 as cv
import torch
import torchvision
import numpy as np


def contour_based_motion_detector(previous_image, next_image, show_bbox=True, contour_area_threshold=100,
                                  video_mode=True):
    diff = cv.absdiff(previous_image, next_image)
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    if show_bbox:
        for contour in contours:
            (x, y, w, h) = cv.boundingRect(contour)
            if cv.contourArea(contour) < contour_area_threshold:
                continue
            cv.rectangle(previous_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv.drawContours(previous_image, contours, -1, (0, 255, 0), 2)

    if not video_mode:
        cv.imshow('motion', previous_image)
        cv.waitKey(0)

        cv.destroyAllWindows()
    return contours


def contour_based_motion_detector_video(video_path, contour_area_threshold=500, background_mode=False):
    cap = cv.VideoCapture(video_path)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    first_frame = copy(frame1)

    while cap.isOpened():
        diff = cv.absdiff(frame1, frame2)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        for contour in contours:
            (x, y, w, h) = cv.boundingRect(contour)
            if cv.contourArea(contour) < contour_area_threshold:
                continue
            if background_mode:
                cv.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if background_mode:
            cv.imshow('motion', frame2)
            frame1 = first_frame
        else:
            cv.imshow('motion', frame1)
            frame1 = frame2

        ret, frame2 = cap.read()

        if cv.waitKey(40) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = '../Datasets/SDD/videos/little/video3/video.mov'
    contour_based_motion_detector_video(path, 500, background_mode=False)
    # prev_image = torchvision.io.read_image('overfit_images/image_t_minus_one.jpeg')
    # next_image = torchvision.io.read_image('overfit_images/image_t_plus_one.jpeg')
    # prev_image, next_image = prev_image.permute(1, 2, 0).numpy(), next_image.permute(1, 2, 0).numpy()
    # contour_based_motion_detector(prev_image, next_image, contour_area_threshold=700, video_mode=False)
