import tensorflow as tf
from utils import *
from config import *
import cv2
import time
import os
import traceback
import math
from PIL import Image

print('inference with {}'.format(INDICES))

# Configure model
before_ch = max(INDICES)
after_ch = max(1, -min(INDICES) + 1)

# Create output directory
output_directory = os.path.join(OUTPUT_DIRECTORY, 'output')
make_dirs(output_directory)

# Open all the videos in video list
video_list = get_videos(TEST_LIST)


sess = tf.Session(
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)))

new_saver = tf.train.import_meta_graph(MODEL_DIRECTORY + MODEL_NAME + '.meta')
new_saver.restore(sess, MODEL_DIRECTORY + MODEL_NAME)
graph = tf.get_default_graph()
x_tensor = graph.get_tensor_by_name('stable_net/input/x_tensor:0')
# output = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_7:0')
# black_pix = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_6:0')
output = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/output_img:0')
black_pix = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/black_pix:0')
# theta_mat_tensor = graph.get_tensor_by_name('stable_net/feature_loss/Reshape:0')
hs_tensor = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/get_Hs/Hs:0')
x_map = graph.get_tensor_by_name("stable_net/inference/SpatialTransformer/_transform/x_map:0")
y_map = graph.get_tensor_by_name("stable_net/inference/SpatialTransformer/_transform/y_map:0")
# black_pix = graph.get_tensor_by_name('stable_net/img_loss/StopGradient:0')


for video_name in video_list:

    # Skip video without name
    if video_name == "":
        continue

    tot_time = 0

    print('Stabilizing: ' + str(video_name))
    stable_video = cv2.VideoCapture(os.path.join(DATA_DIRECTORY, 'stable', video_name))
    unstable_video = cv2.VideoCapture(os.path.join(DATA_DIRECTORY, 'unstable', video_name))

    fps = unstable_video.get(cv2.CAP_PROP_FPS)
    print("fps:" + str(fps))

    videoWriter = cv2.VideoWriter(os.path.join(output_directory, video_name),
                                  cv2.VideoWriter_fourcc(*'mp4v'), fps, (WIDTH, HEIGHT))
    before_frames = []
    before_masks = []
    after_frames = []
    after_temp = []

    # Read first frame
    valid, stable_video_frame = stable_video.read()
    valid, unstable_video_frame = unstable_video.read()

    if START_WITH_STABLE:
        frame = stable_video_frame
    else:
        frame = unstable_video_frame

    # Scale clip if necessary
    videoWriter.write(cv2.resize(frame, (WIDTH, HEIGHT)))

    for i in range(before_ch):
        before_frames.append(cvt_img2train(frame, CROP_RATE))
        before_masks.append(np.zeros([1, HEIGHT, WIDTH, 1], dtype=np.float))

        '''
        temp = before_frames[i]
        temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)
        temp = np.concatenate([temp, np.zeros_like(temp)], axis=1)
        temp = np.concatenate([temp, np.zeros_like(temp)], axis=0)
        '''

    # for i in range(after_ch + 1):
    for i in range(after_ch):
        valid, frame = unstable_video.read()
        frame_unstable = frame
        after_temp.append(frame)
        after_frames.append(cvt_img2train(frame, 1))

    length = 0
    in_xs = []
    delta = 0
    speed = RANDOM_BLACK
    dh = int(HEIGHT * 0.8 / 2)
    dw = int(WIDTH * 0.8 / 2)
    all_black = np.zeros([HEIGHT, WIDTH], dtype=np.int64)
    frames = []

    black_mask = np.zeros([dh, WIDTH], dtype=np.float)
    temp_mask = np.concatenate(
        [np.zeros([HEIGHT - 2 * dh, dw], dtype=np.float), np.ones([HEIGHT - 2 * dh, WIDTH - 2 * dw], dtype=np.float),
         np.zeros([HEIGHT - 2 * dh, dw], dtype=np.float)], axis=1)
    black_mask = np.reshape(np.concatenate([black_mask, temp_mask, black_mask], axis=0), [1, HEIGHT, WIDTH, 1])

    name = 0
    while True:

        in_x = []

        if INPUT_MASK:
            for i in INDICES:
                if i > 0:
                    in_x.append(before_masks[-i])

        for i in INDICES:
            if i > 0:
                in_x.append(before_frames[-i])

        in_x.append(after_frames[0])

        for i in INDICES:
            if i < 0:
                in_x.append(after_frames[-i])

        if NO_BM == 0:
            in_x.append(black_mask)

        # for i in range(after_ch + 1):
        in_x = np.concatenate(in_x, axis=3)

        # for max span
        if MAX_SPAN != 1:
            in_xs.append(in_x)
            if len(in_xs) > MAX_SPAN:
                in_xs = in_xs[-1:]
                print('cut')
            in_x = in_xs[0].copy()
            in_x[0, ..., before_ch] = after_frames[0][..., 0]

        tmp_in_x = in_x.copy()

        for j in range(REFINE):
            start = time.time()
            img, black, Hs, x_map_, y_map_ = sess.run([output, black_pix, hs_tensor, x_map, y_map],
                                                      feed_dict={x_tensor: tmp_in_x})

            tot_time += time.time() - start
            black = black[0, :, :]

            mask_image = np.asarray(black, np.uint8)
            mask_image *= int(255 / mask_image.max())
            mask_image = Image.fromarray(mask_image)
            mask_image.save('masks/' + str(name) + '.jpg')
            name += 1

            xmap = x_map_[0, :, :, 0]
            ymap = y_map_[0, :, :, 0]
            all_black = all_black + np.round(black).astype(np.int64)
            img = img[0, :, :, :].reshape(HEIGHT, WIDTH)
            frame = img + black * (-1)
            frame = frame.reshape(1, HEIGHT, WIDTH, 1)
            tmp_in_x[..., -1] = frame[..., 0]

        img = ((np.reshape(img, (HEIGHT, WIDTH)) + 0.5) * 255).astype(np.uint8)

        net_output = img
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # img_warped = warpRevBundle(cv2.resize(frame_unstable, (width, height)), Hs[0])
        img_warped = warp_rev_bundle2(cv2.resize(after_temp[0], (WIDTH, HEIGHT)), xmap, ymap)

        frame_img = Image.fromarray(img_warped)
        frame_img.save('masks/' + str(name) + '_frame.jpg')

        frames.append(img_warped)
        videoWriter.write(img_warped)

        valid, frame_unstable = unstable_video.read()

        if not valid:
            break
        length = length + 1
        if length % 10 == 0:
            print("length: " + str(length))
            print('fps: {}'.format(length / tot_time))

        before_frames.append(frame)
        before_masks.append(black.reshape((1, HEIGHT, WIDTH, 1)))

        if INFER_WITH_LAST:
            for i in range(len(before_frames)):
                before_frames[i] = before_frames[-1]

        before_frames.pop(0)
        before_masks.pop(0)
        after_frames.append(cvt_img2train(frame_unstable, 1))
        after_frames.pop(0)
        after_temp.append(frame_unstable)
        after_temp.pop(0)

        # if (len == 100):
        #    break

    print('total length={}'.format(length + 2))
    videoWriter.release()
    unstable_video.release()

    # Create black sum
    black_sum = np.zeros([HEIGHT + 1, WIDTH + 1], dtype=np.int64)
    for row in range(HEIGHT):
        for column in range(WIDTH):
            black_sum[row + 1][column + 1] = black_sum[row][column + 1] + black_sum[row + 1][column] \
                                             - black_sum[row][column] + all_black[row][column]

    max_span = 0
    crop_mask = []

    for row in range(0, int(math.floor(HEIGHT * 0.5)), 10):
        for column in range(0, int(math.floor(WIDTH * 0.5)), 10):

            # If not in a black region
            if all_black[row][column] > 0:
                continue

            for hh in range(row, HEIGHT):
                for ww in range(column, WIDTH):

                    if black_sum[hh + 1][ww + 1] - black_sum[hh + 1][column] \
                            - black_sum[row][ww + 1] + black_sum[row][column] > 0:
                        break

                    else:
                        span = (hh - row + 1) * (ww - column + 1)
                        if span > max_span:
                            max_span = span
                            crop_mask = [row, column, hh, ww]

    videoWriter_cut = cv2.VideoWriter(os.path.join(output_directory, 'cut_' + video_name),
                                      cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                      (crop_mask[3] - crop_mask[1] + 1, crop_mask[2] - crop_mask[0] + 1))

    for frame in frames:
        frame_ = frame[crop_mask[0]:crop_mask[2] + 1, crop_mask[1]:crop_mask[3] + 1, :]
        videoWriter_cut.write(frame_)
    videoWriter_cut.release()
