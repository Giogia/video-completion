from config import *
import cv2
import os
import math
from PIL import Image
import numpy as np


def get_videos(test_list):

    video_list = []

    if os.path.isfile(test_list):
        print('Adding ' + test_list)
        list_file = open(test_list, 'r').read()
        video_list.extend(list_file.split('\n'))

    return video_list


def make_dirs(path):

    if not os.path.exists(path):
        os.makedirs(path)


def cvt_img2train(img, cropping_rate=1):

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    if cropping_rate != 1:
        h = int(HEIGHT / cropping_rate)
        dh = int((h - HEIGHT) / 2)
        w = int(WIDTH / cropping_rate)
        dw = int((w - WIDTH) / 2)

        img = img.resize((w, h), Image.BILINEAR)
        img = img.crop((dw, dh, dw + WIDTH, dh + HEIGHT))

    else:
        img = img.resize((WIDTH, HEIGHT), Image.BILINEAR)

    img = np.array(img)
    img = img * (1. / 255) - 0.5
    img = img.reshape((1, HEIGHT, WIDTH, 1))

    return img


def cvt_train2img(x):
    return ((np.reshape(x, (HEIGHT, WIDTH)) + 0.5) * 255).astype(np.uint8)


def cvt2int32(x):
    return x.astype(np.int32)


def draw_imgs(net_output, stable_frame, unstable_frame, inputs):

    assert (net_output.ndim == 2)
    assert (stable_frame.ndim == 2)
    assert (unstable_frame.ndim == 2)

    net_output = cvt2int32(net_output)
    stable_frame = cvt2int32(stable_frame)
    unstable_frame = cvt2int32(unstable_frame)
    last_frame = cvt2int32(cvt_train2img(inputs[..., 0]))
    output_minus_input = abs(net_output - unstable_frame)
    output_minus_stable = abs(net_output - stable_frame)
    output_minus_last = abs(net_output - last_frame)
    img_top = np.concatenate([net_output, output_minus_stable], axis=1)
    img_bottom = np.concatenate([output_minus_input, output_minus_last], axis=1)
    img = np.concatenate([img_top, img_bottom], axis=0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def get_next(delta, bound, speed=5):
    tmp = delta + speed
    if tmp >= bound or tmp < 0: speed *= -1
    return delta + speed, speed
    # return np.random.randint(0, bound), 5


def cvt_theta_mat(theta_mat):
    # theta_mat * x = x'
    # ret * scale_mat * x = scale_mat * x'
    # ret = scale_mat * theta_mat * scale_mat^-1
    scale_mat = np.eye(3)
    scale_mat[0, 0] = WIDTH / 2.
    scale_mat[0, 2] = WIDTH / 2.
    scale_mat[1, 1] = HEIGHT / 2.
    scale_mat[1, 2] = HEIGHT / 2.
    assert (theta_mat.shape == (3, 3))
    from numpy.linalg import inv
    return np.matmul(np.matmul(scale_mat, theta_mat), inv(scale_mat))


def warp_rev(img, theta):
    assert (img.ndim == 3)
    assert (img.shape[-1] == 3)
    theta_mat_cvt = cvt_theta_mat(theta)
    return cv2.warpPerspective(img, theta_mat_cvt, dsize=(WIDTH, HEIGHT), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def cvt_theta_mat_bundle(hs):
    # theta_mat * x = x'
    # ret * scale_mat * x = scale_mat * x'
    # ret = scale_mat * theta_mat * scale_mat^-1
    scale_mat = np.eye(3)
    scale_mat[0, 0] = WIDTH / 2.
    scale_mat[0, 2] = WIDTH / 2.
    scale_mat[1, 1] = HEIGHT / 2.
    scale_mat[1, 2] = HEIGHT / 2.

    hs = hs.reshape((GRID_H, GRID_W, 3, 3))
    from numpy.linalg import inv

    return np.matmul(np.matmul(scale_mat, hs), inv(scale_mat))


def warp_rev_bundle2(img, x_map, y_map):
    assert (img.ndim == 3)
    assert (img.shape[-1] == 3)
    rate = 4
    x_map = cv2.resize(cv2.resize(x_map, (int(WIDTH / rate), int(HEIGHT / rate))), (WIDTH, HEIGHT))
    y_map = cv2.resize(cv2.resize(y_map, (int(WIDTH / rate), int(HEIGHT / rate))), (WIDTH, HEIGHT))
    x_map = (x_map + 1) / 2 * WIDTH
    y_map = (y_map + 1) / 2 * HEIGHT
    dst = cv2.remap(img, x_map, y_map, cv2.INTER_LINEAR)
    assert (dst.shape == (HEIGHT, WIDTH, 3))
    return dst


def warp_rev_bundle(img, hs):
    assert (img.ndim == 3)
    assert (img.shape[-1] == 3)
    hs_cvt = cvt_theta_mat_bundle(hs)

    gh = int(math.floor(HEIGHT / GRID_H))
    gw = int(math.floor(WIDTH / GRID_W))
    img_ = []
    for i in range(GRID_H):
        row_img_ = []
        for j in range(GRID_W):
            h = hs_cvt[i, j, :, :]
            sh = i * gh
            eh = (i + 1) * gh - 1
            sw = j * gw
            ew = (j + 1) * gw - 1
            if i == GRID_H - 1:
                eh = HEIGHT - 1
            if j == GRID_W - 1:
                ew = WIDTH - 1
            temp = cv2.warpPerspective(img, h, dsize=(WIDTH, HEIGHT), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            row_img_.append(temp[sh:eh + 1, sw:ew + 1, :])
        img_.append(np.concatenate(row_img_, axis=1))
    img = np.concatenate(img_, axis=0)
    assert (img.shape == (HEIGHT, WIDTH, 3))
    return img

