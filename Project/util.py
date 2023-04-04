import numpy as np
import cv2
import torch


def f(Para, reflection_matrix):
    '''
        解非线性方程组
        通过仿射变换矩阵来近似提取其中的平移,旋转,拉伸等运动
    '''
    a = Para[0]
    b = Para[1]
    c = Para[2]
    d = Para[3]
    e = Para[4]
    return [(a+c) - reflection_matrix[0, 0],
            (b + d) - reflection_matrix[0, 1],
            (-b + d) - reflection_matrix[1, 0],
            (a + e) - reflection_matrix[1, 1],
            a**2 + b**2 - 1]


def rand_int_3():
    '''
        返回三个颜色的随机数
    '''
    result = []
    result.append(np.random.randint(0, 255))
    result.append(np.random.randint(0, 255))
    result.append(np.random.randint(0, 255))
    return result


def write_feature(curr, prev_corners, curr_corners, feature_out):
    '''
        输出一个视频的光流运动视频
    '''
    for (new, old) in zip(prev_corners, curr_corners):
        a, b = new.ravel()
        c, d = old.ravel()
        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)
        curr = cv2.line(curr, (a, b), (c, d), rand_int_3(), 3)
        cv2.circle(curr, (a, b), 2, rand_int_3(), - 1)
    feature_out.write(curr)


def gaussian_pattern(K_size, sigma=0.8):
    '''
        生成一维高斯滤波核
    '''
    if K_size % 2 == 0:
        print("The size is wrong")
        return
    k = K_size//2
    matrix_ans = np.zeros((K_size), dtype=float)

    for i in range(-k, k):
        matrix_ans[k+i] = np.exp(-i ** 2/(2*sigma ** 2))
        matrix_ans[k+i] /= sigma * np.sqrt(2*np.pi)
    matrix_ans /= matrix_ans.sum()  # 归一化处理
    return matrix_ans


def filter_avg(curve, ksize):
    '''
        均值滤波
    '''
    window_size = 2 * ksize + 1
    # 定义过滤器
    f = np.ones(window_size) / window_size
    # 为边界添加填充
    curve_pad = np.lib.pad(curve, (ksize, ksize), 'edge')
    # 应用卷积
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # 删除填充
    curve_smoothed = curve_smoothed[ksize:-ksize]
    # 返回平滑曲线
    return curve_smoothed


def filter_gausain(curve, ksize):
    '''
        高斯滤波
    '''
    window_size = 2 * ksize + 1
    # 定义过滤器
    f = gaussian_pattern(window_size)
    # 为边界添加填充
    curve_pad = np.lib.pad(curve, (ksize, ksize), 'edge')
    # 应用卷积
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # 删除填充
    curve_smoothed = curve_smoothed[ksize:-ksize]
    # 返回平滑曲线
    return curve_smoothed


def smooth_f(total_track,size1, size2, size3):
    '''
        平滑轨迹
    '''
    smoothed_tack = np.copy(total_track)
    for i in range(6):
        smoothed_tack[:, i] = filter_avg(
            smoothed_tack[:, i], ksize=size1)
        smoothed_tack[:, i] = filter_gausain(
            smoothed_tack[:, i], ksize=size2)
        smoothed_tack[:, i] = filter_avg(
            smoothed_tack[:, i], ksize=size3)
    return smoothed_tack


def fixBorder(frame):
    '''
        适应屏幕
    '''
    s = frame.shape
    # 在不移动中心的情况下，将图像缩放7%
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.07)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame





def d(P, C, size_x, size_y):
    '''
        计算两个点阵的d值
        实际意义计算两个相邻帧的角点的晃动程度
    '''
    x = P[:,  0] - C[:, 0]
    y = P[:,  1] - C[:, 1]
    x = x * x
    y = y * y
    ans = np.sum(x + y)/(size_x * size_y)
    ans = np.sqrt(ans)
    ans = ans / (np.sqrt(2) * C.shape[1])
    return ans


def tensor_d(P, C, size_x, size_y):
    '''
        用于tensor的数据计算T值, 为了梯度下降
    '''
    x = P[:,  0] - C[:, 0]
    y = P[:,  1] - C[:, 1]
    x = x * x
    y = y * y
    ans = torch.sum(x + y)/(size_x * size_y)
    ans = torch.sqrt(ans)
    ans = ans / (np.sqrt(2) * C.shape[1])
    return ans


def function(x, src_points, des_points, w, h):
    '''
        梯度下降的函数
    '''
    src_points = torch.from_numpy(src_points)
    src_points = src_points.to(torch.double)
    des_points = torch.from_numpy(des_points)
    des_points = des_points.to(torch.double)
    f = torch.mm(src_points, x)
    f = tensor_d(f, des_points, w, h)
    return f


def get_d_a_video(filename):
    '''
        获取一个视频的平均D值
        实际用于得到处理后的的D值
    '''
    dn = []
    capture = cv2.VideoCapture(filename)
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    _, prev = capture.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i in range(1, n_frames):
        prev_corners = cv2.goodFeaturesToTrack(prev_gray,
                                               maxCorners=200,
                                               qualityLevel=0.03,
                                               minDistance=30,
                                               blockSize=3
                                               )
        ret, curr = capture.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_corners, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_corners, None)
        prev_corners = prev_corners[status == 1]
        curr_corners = curr_corners[status == 1]
        d_temp = d(prev_corners, curr_corners, w, h)
        dn.append(d_temp)
        prev = curr
        prev_gray = curr_gray

    dn = np.array(dn)
    dn = np.mean(dn)
    return dn
