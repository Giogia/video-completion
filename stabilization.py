import tensorflow as tf
from utils import *
import cv2
import time
import os
import traceback
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', default='models')
parser.add_argument('--model-name', default='model-90000')
parser.add_argument('--before-ch', type=int)
# parser.add_argument('--after-ch', type=int)
parser.add_argument('--output-dir', default='output')
parser.add_argument('--infer-with-stable', action='store_true')
parser.add_argument('--infer-with-last', action='store_true')
parser.add_argument('--test-list', nargs='+', default=['video-list.txt', 'train-video-list.txt'])
# parser.add_argument('--train-list', default='data_video/train_list_deploy')
parser.add_argument('--prefix', default='data')
parser.add_argument('--max-span', type=int, default=1)
parser.add_argument('--random-black', type=int, default=None)
# parser.add_argument('--indices', type=int, nargs='+', required=True)
parser.add_argument('--start-with-stable', action='store_true')
parser.add_argument('--refine', type=int, default=1)
parser.add_argument('--no_bm', type=int, default=1)
parser.add_argument('--gpu_memory_fraction', type=float, default=0.9)
parser.add_argument('--deploy-vis', action='store_true')
args = parser.parse_args()

MaxSpan = args.max_span
args.indices = indices[1:]

sess = tf.Session(
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)))

model_dir = args.model_dir  # 'models/vbeta-1.1.0/'
model_name = args.model_name  # 'model-5000'
before_ch = max(args.indices)  # args.before_ch
after_ch = max(1, -min(args.indices) + 1)
# after_ch = args.after_ch
# after_ch = 0
new_saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
new_saver.restore(sess, model_dir + model_name)
graph = tf.get_default_graph()
x_tensor = graph.get_tensor_by_name('stable_net/input/x_tensor:0')
# output = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_7:0')
# black_pix = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_6:0')
output = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/output_img:0')
black_pix = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/black_pix:0')
# theta_mat_tensor = graph.get_tensor_by_name('stable_net/feature_loss/Reshape:0')
Hs_tensor = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/get_Hs/Hs:0')
x_map = graph.get_tensor_by_name("stable_net/inference/SpatialTransformer/_transform/x_map:0")
y_map = graph.get_tensor_by_name("stable_net/inference/SpatialTransformer/_transform/y_map:0")

# black_pix = graph.get_tensor_by_name('stable_net/img_loss/StopGradient:0')


video_list = []

for list_path in args.test_list:
    if os.path.isfile(list_path):
        print('adding ' + list_path)
        list_f = open(list_path, 'r')
        temp = list_f.read()
        video_list.extend(temp.split('\n'))

production_dir = os.path.join(args.output_dir, 'output')
visual_dir = os.path.join(args.output_dir, 'output-vis')
make_dirs(production_dir)
make_dirs(visual_dir)

print('inference with {}'.format(args.indices))
for video_name in video_list:
    tot_time = 0
    if video_name == "":
        continue
    print(video_name)
    stable_cap = cv2.VideoCapture(os.path.join(args.prefix, 'stable', video_name))
    unstable_cap = cv2.VideoCapture(os.path.join(args.prefix, 'unstable', video_name))
    fps = unstable_cap.get(cv2.CAP_PROP_FPS)
    cut_fps = False
    if fps > 40:
        fps /= 2
        cut_fps = True
    print("fps:" + str(fps))
    print(os.path.join(args.prefix, 'unstable', video_name))
    videoWriter = cv2.VideoWriter(os.path.join(production_dir, video_name),
                                  cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if args.deploy_vis:
        videoWriterVis = cv2.VideoWriter(os.path.join(visual_dir, video_name),
                                         cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height * 2))
    before_frames = []
    before_masks = []
    after_frames = []
    after_temp = []

    ret, stable_cap_frame = stable_cap.read()
    ret, unstable_cap_frame = unstable_cap.read()
    if args.start_with_stable:
        frame = stable_cap_frame
    else:
        frame = unstable_cap_frame
    videoWriter.write(cv2.resize(frame, (width, height)))
    for i in range(before_ch):
        before_frames.append(cvt_img2train(frame, crop_rate))
        before_masks.append(np.zeros([1, height, width, 1], dtype=np.float))
        temp = before_frames[i]
        temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)
        # videoWriter.write(cv2.resize(stable_cap_frame, (width, height)))
        temp = np.concatenate([temp, np.zeros_like(temp)], axis=1)
        temp = np.concatenate([temp, np.zeros_like(temp)], axis=0)
        if args.deploy_vis:
            videoWriterVis.write(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))

    # for i in range(after_ch + 1):
    for i in range(after_ch):
        if cut_fps:
            ret, frame = unstable_cap.read()
        ret, frame = unstable_cap.read()
        frame_unstable = frame
        after_temp.append(frame)
        after_frames.append(cvt_img2train(frame, 1))

    length = 0
    in_xs = []
    delta = 0
    speed = args.random_black
    dh = int(height * 0.8 / 2)
    dw = int(width * 0.8 / 2)
    all_black = np.zeros([height, width], dtype=np.int64)
    frames = []

    black_mask = np.zeros([dh, width], dtype=np.float)
    temp_mask = np.concatenate(
        [np.zeros([height - 2 * dh, dw], dtype=np.float), np.ones([height - 2 * dh, width - 2 * dw], dtype=np.float),
         np.zeros([height - 2 * dh, dw], dtype=np.float)], axis=1)
    black_mask = np.reshape(np.concatenate([black_mask, temp_mask, black_mask], axis=0), [1, height, width, 1])

    try:
        while True:
            if args.deploy_vis:
                _, stable_cap_frame = stable_cap.read()
                stable_train_frame = cvt_img2train(stable_cap_frame, crop_rate)
                if args.random_black is not None:
                    delta, speed = get_next(delta, 50, speed)
                    print(delta, speed)
                    stable_train_frame[:, :, delta:width, ...] = stable_train_frame[:, :, 0:width - delta, ...]
                    stable_train_frame[:, :, :delta, ...] = -1
                stable_frame = cvt_train2img(stable_train_frame)
                unstable_frame = cvt_train2img(after_frames[0])
            in_x = []
            if input_mask:
                for i in args.indices:
                    if i > 0:
                        in_x.append(before_masks[-i])
            for i in args.indices:
                if i > 0:
                    in_x.append(before_frames[-i])
            in_x.append(after_frames[0])
            for i in args.indices:
                if i < 0:
                    in_x.append(after_frames[-i])
            if args.no_bm == 0:
                in_x.append(black_mask)
            # for i in range(after_ch + 1):
            in_x = np.concatenate(in_x, axis=3)
            # for max span
            if MaxSpan != 1:
                in_xs.append(in_x)
                if len(in_xs) > MaxSpan:
                    in_xs = in_xs[-1:]
                    print('cut')
                in_x = in_xs[0].copy()
                in_x[0, ..., before_ch] = after_frames[0][..., 0]
            tmp_in_x = in_x.copy()
            for j in range(args.refine):
                start = time.time()
                img, black, Hs, x_map_, y_map_ = sess.run([output, black_pix, Hs_tensor, x_map, y_map],
                                                          feed_dict={x_tensor: tmp_in_x})
                tot_time += time.time() - start
                black = black[0, :, :]
                xmap = x_map_[0, :, :, 0]
                ymap = y_map_[0, :, :, 0]
                all_black = all_black + np.round(black).astype(np.int64)
                img = img[0, :, :, :].reshape(height, width)
                frame = img + black * (-1)
                frame = frame.reshape(1, height, width, 1)
                tmp_in_x[..., -1] = frame[..., 0]
            img = ((np.reshape(img, (height, width)) + 0.5) * 255).astype(np.uint8)

            net_output = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # img_warped = warpRevBundle(cv2.resize(frame_unstable, (width, height)), Hs[0])
            img_warped = warp_rev_bundle2(cv2.resize(after_temp[0], (width, height)), xmap, ymap)
            frames.append(img_warped)
            videoWriter.write(img_warped)
            if args.deploy_vis:
                videoWriterVis.write(draw_imgs(net_output, stable_frame, unstable_frame, in_x))

            if cut_fps:
                ret, frame_unstable = unstable_cap.read()
            ret, frame_unstable = unstable_cap.read()

            if not ret:
                break
            length = length + 1
            if length % 10 == 0:
                print("length: " + str(length))
                print('fps: {}'.format(length / tot_time))
            if args.infer_with_stable:
                before_frames.append(stable_train_frame)
            else:
                before_frames.append(frame)
                before_masks.append(black.reshape((1, height, width, 1)))
            if args.infer_with_last:
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
    except Exception as e:
        traceback.print_exc()
    finally:
        print('total length={}'.format(length + 2))
        videoWriter.release()
        unstable_cap.release()

        black_sum = np.zeros([height + 1, width + 1], dtype=np.int64)
        for i in range(height):
            for j in range(width):
                black_sum[i + 1][j + 1] = black_sum[i][j + 1] + black_sum[i + 1][j] - black_sum[i][j] + all_black[i][j]
        max_s = 0
        ans = []
        for i in range(0, int(math.floor(height * 0.5)), 10):
            print(i)
            print(max_s)
            for j in range(0, int(math.floor(width * 0.5)), 10):
                if all_black[i][j] > 0:
                    continue
                for hh in range(i, height):
                    dw = int(math.floor(float(max_s) / (hh - i + 1)))
                    for ww in range(j, width):
                        if black_sum[hh + 1][ww + 1] - black_sum[hh + 1][j] \
                                - black_sum[i][ww + 1] + black_sum[i][j] > 0:
                            break
                        else:
                            s = (hh - i + 1) * (ww - j + 1)
                            if s > max_s:
                                max_s = s
                                ans = [i, j, hh, ww]
        videoWriter_cut = cv2.VideoWriter(os.path.join(production_dir, 'cut_' + video_name),
                                          cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                          (ans[3] - ans[1] + 1, ans[2] - ans[0] + 1))
        for frame in frames:
            frame_ = frame[ans[0]:ans[2] + 1, ans[1]:ans[3] + 1, :]
            videoWriter_cut.write(frame_)
        videoWriter_cut.release()
