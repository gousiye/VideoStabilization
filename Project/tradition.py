import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from scipy.optimize import fsolve
import util
import os


def tradition_stabilization(input_video, outpath):
    (inputpath, file) = os.path.split(input_video)
    (filename, suffix) = os.path.splitext(file)
    #print(outpath, filename, suffix)
    outpath = outpath + filename + '/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    feature_video = outpath + filename + '_feature.mp4'  # 对视频的角点和光流的提取
    output_video = outpath + filename + '_out.mp4'
    compare_video = outpath + filename + '_compare.mp4'
    capture = cv2.VideoCapture(input_video)

    # 计算前后两个视频的平s均晃动水平
    dn_original = []
    dn_processed = []

    # 帧的总数，视频中图像的长，宽，帧率，编码格式的设置(mp4)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 用来记录仿射变换变换矩阵代表的运动轨迹
    affine_matrix = np.zeros((n_frames, 6)).tolist()

    # 定义输出形式
    feature_out = cv2.VideoWriter(feature_video, fourcc, fps, (w, h))
    result_out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    compare_out = cv2.VideoWriter(compare_video, fourcc, fps, (2*w, h))
    # 平均，高斯滤波核的大小
    SMOOTHSIZE1 = 50
    SMOOTHSIZE2 = 25
    SMOOTHSIZE3 = 35

    # 读取第一帧并转为灰度图
    _, prev = capture.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for i in range(1, n_frames):
        # 提取角点，返回角点的坐标
        prev_corners = cv2.goodFeaturesToTrack(prev_gray,
                                               maxCorners=200,
                                               qualityLevel=0.03,
                                               minDistance=30,
                                               blockSize=3
                                               )
    # print(prev_corners.dtype)

        # 读取下一帧，然后与当前帧进行光流计算
        success, curr = capture.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_corners, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_corners, None)
        #print(curr_corners.shape, prev_corners.shape)
        # 提取出光流找到的有效点
        prev_corners = prev_corners[status == 1]
        curr_corners = curr_corners[status == 1]

        # 计算晃动程度
        d_temp = util.d(prev_corners, curr_corners, w, h)
        dn_original.append(d_temp)

        util.write_feature(curr, prev_corners, curr_corners, feature_out)
        # 设置进度条
        print("\r", end="")
        print("Get Features of " + file + ": {}%: ".format(
            100*(i + 1)//n_frames), "▋" * (100*(i + 1) // n_frames), end="")
        #print(prev_corners.shape, curr_corners.shape)
        # 基于RANSC求得二维仿射变换矩阵
        reflection_matrix, inlier = cv2.estimateAffine2D(
            prev_corners, curr_corners,)

        # 从仿射变换中提取变换元素
        # x = a0x + a1y + a2
        # y = b0x + b1y + b2

        # 提取平移运算
        if(np.shape(reflection_matrix) != (2, 3)):
            continue
        dmove_x = reflection_matrix[0, 2]
        dmove_y = reflection_matrix[1, 2]
        # 模拟提取旋转，拉伸等元素
        rest = fsolve(util.f, [0, 0, 0, 0, 0], reflection_matrix)
        drotation = np.arctan(rest[1]/rest[0])  # 旋转
        #drotation = np.arctan2(reflection_matrix[1, 0], reflection_matrix[0, 0])
        dscale = rest[2]  # 拉伸
        dstretching = rest[3]  # 拉伸
        drest = rest[4]
        affine_matrix[i] = [dmove_x, dmove_y,
                            drotation, dscale, dstretching, drest]
        # 移动到下一帧
        prev = curr
        prev_gray = curr_gray

    # 记录的运动轨迹
    total_track = np.cumsum(affine_matrix, axis=0)
    smoothed_tack = util.smooth_f(
        total_track, SMOOTHSIZE1, SMOOTHSIZE2, SMOOTHSIZE3)
    difference = smoothed_tack - total_track
    affine_smooth = affine_matrix + difference
    # 将视频重置到第一帧，准备写入平均后的运动轨迹
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print()
    # 读取第一帧
    for i in range(0, n_frames-1):
        # 读下一帧
        success, frame = capture.read()
        if not success:
            break
        # 将运动轨迹在转换为仿射矩阵
        ref_m = np.zeros((2, 3), np.float32)
        ref_m[0, 0] = np.cos(affine_smooth[i, 2]) + affine_smooth[i, 3]
        ref_m[0, 1] = np.sin(affine_smooth[i, 2]) + affine_smooth[i, 4]
        ref_m[1, 0] = -np.sin(affine_smooth[i, 2]) + affine_smooth[i, 4]
        ref_m[1, 1] = np.cos(affine_smooth[i, 2]) + affine_smooth[i, 5]
        ref_m[0, 2] = affine_smooth[i, 0]
        ref_m[1, 2] = affine_smooth[i, 1]
        #frame_copy = frame.copy()
        #frame_copy = cv2.resize(frame_copy, (w_out, h_out))

        frame_stable = cv2.warpAffine(frame, ref_m, (w, h))
        frame_stable = util.fixBorder(frame_stable)

        print("\r", end="")
        print("Stabilizing " + file + ": {}%: ".format(
            100*(i + 1)//n_frames), "▋" * (100*(i + 1) // n_frames), end="")
        result_out.write(frame_stable)

        frame_out = cv2.hconcat([frame, frame_stable])

        compare_out.write(frame_out)

    # 最后一帧不用处理
    sucess, frame = capture.read()
    frame = util.fixBorder(frame)
    print("\r", end="")
    i = n_frames - 1
    print("Stabilizing " + file + ": {}%: ".format(
        100*(i + 1)//n_frames), "▋" * (100*(i + 1) // n_frames), end="")
    result_out.write(frame)
    frame_out = cv2.hconcat([frame, frame])
    compare_out.write(frame_out)

    feature_out.release()
    result_out.release()
    compare_out.release()

    # 计算所有dn的平均值
    dn_original = np.array(dn_original)
    dn_original = np.mean(dn_original)
    dn_processed = util.get_d_a_video(output_video)
    print()
    print()
    return [dn_original, dn_processed]
