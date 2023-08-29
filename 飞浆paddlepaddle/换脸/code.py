# 导入相关数据包
import cv2
import numpy as np
import paddlehub as hub
from moviepy.editor import *
import shutil, os
# 人脸检测 换脸函数
def get_image_size(image):
    """
    获取图片大小（高度,宽度）
    :param image: image
    :return: （高度,宽度）
    """
    image_size = (image.shape[0], image.shape[1])
    return image_size


def get_face_landmarks(image):
    """
    获取人脸标志，68个特征点
    :param image: image
    :param face_detector: dlib.get_frontal_face_detector
    :param shape_predictor: dlib.shape_predictor
    :return: np.array([[],[]]), 68个特征点
    """
    dets = face_landmark.keypoint_detection([image])
    #num_faces = len(dets[0]['data'][0])
    if len(dets) == 0:
        print("Sorry, there were no faces found.")
        return None
    # shape = shape_predictor(image, dets[0])
    face_landmarks = np.array([[p[0], p[1]] for p in dets[0]['data'][0]])
    return face_landmarks


def get_face_mask(image_size, face_landmarks):
    """
    获取人脸掩模
    :param image_size: 图片大小
    :param face_landmarks: 68个特征点
    :return: image_mask, 掩模图片
    """
    mask = np.zeros(image_size, dtype=np.int32)
    points = np.concatenate([face_landmarks[0:16], face_landmarks[26:17:-1]])
    points = np.array(points, dtype=np.int32)

    cv2.fillPoly(img=mask, pts=[points], color=255)

    # mask = np.zeros(image_size, dtype=np.uint8)
    # points = cv2.convexHull(face_landmarks)  # 凸包
    # cv2.fillConvexPoly(mask, points, color=255)
    return mask.astype(np.uint8)


def get_affine_image(image1, image2, face_landmarks1, face_landmarks2):
    """
    获取图片1仿射变换后的图片
    :param image1: 图片1, 要进行仿射变换的图片
    :param image2: 图片2, 只要用来获取图片大小，生成与之大小相同的仿射变换图片
    :param face_landmarks1: 图片1的人脸特征点
    :param face_landmarks2: 图片2的人脸特征点
    :return: 仿射变换后的图片
    """
    three_points_index = [18, 8, 25]
    M = cv2.getAffineTransform(face_landmarks1[three_points_index].astype(np.float32),
                               face_landmarks2[three_points_index].astype(np.float32))
    dsize = (image2.shape[1], image2.shape[0])
    affine_image = cv2.warpAffine(image1, M, dsize)
    return affine_image.astype(np.uint8)


def get_mask_center_point(image_mask):
    """
    获取掩模的中心点坐标
    :param image_mask: 掩模图片
    :return: 掩模中心
    """
    image_mask_index = np.argwhere(image_mask > 0)
    miny, minx = np.min(image_mask_index, axis=0)
    maxy, maxx = np.max(image_mask_index, axis=0)
    center_point = ((maxx + minx) // 2, (maxy + miny) // 2)
    return center_point


def get_mask_union(mask1, mask2):
    """
    获取两个掩模掩盖部分的并集
    :param mask1: mask_image, 掩模1
    :param mask2: mask_image, 掩模2
    :return: 两个掩模掩盖部分的并集
    """
    mask = np.min([mask1, mask2], axis=0)  # 掩盖部分并集
    mask = ((cv2.blur(mask, (5, 5)) == 255) * 255).astype(np.uint8)  # 缩小掩模大小
    mask = cv2.blur(mask, (3, 3)).astype(np.uint8)  # 模糊掩模
    return mask


def skin_color_adjustment(im1, im2, mask=None):
    """
    肤色调整
    :param im1: 图片1
    :param im2: 图片2
    :param mask: 人脸 mask. 如果存在，使用人脸部分均值来求肤色变换系数；否则，使用高斯模糊来求肤色变换系数
    :return: 根据图片2的颜色调整的图片1
    """
    if mask is None:
        im1_ksize = 55
        im2_ksize = 55
        im1_factor = cv2.GaussianBlur(im1, (im1_ksize, im1_ksize), 0).astype(np.float)
        im2_factor = cv2.GaussianBlur(im2, (im2_ksize, im2_ksize), 0).astype(np.float)
    else:
        im1_face_image = cv2.bitwise_and(im1, im1, mask=mask)
        im2_face_image = cv2.bitwise_and(im2, im2, mask=mask)
        im1_factor = np.mean(im1_face_image, axis=(0, 1))
        im2_factor = np.mean(im2_face_image, axis=(0, 1))

    im1 = np.clip((im1.astype(np.float) * im2_factor / np.clip(im1_factor, 1e-6, None)), 0, 255).astype(np.uint8)
    return im1


def change_face(im_name1, im_name2, new_path):
    """
    :param im1: 要替换成的人脸图片或文件名
    :param im_name2: 原始人脸图片文件名
    :param new_path: 替换后图片目录
    """
    if isinstance(im_name1, str):
        im1 = cv2.imread(im_name1)  # face_image
    else:
        im1 = im_name1
    im1 = cv2.resize(im1, (600, im1.shape[0] * 600 // im1.shape[1]))
    landmarks1 = get_face_landmarks(im1)  # 68_face_landmarks

    im1_size = get_image_size(im1)  # 脸图大小
    im1_mask = get_face_mask(im1_size, landmarks1)  # 脸图人脸掩模

    im2 = cv2.imread(im_name2)
    landmarks2 = get_face_landmarks(im2)  # 68_face_landmarks
    if landmarks2 is not None:
        im2_size = get_image_size(im2)  # 摄像头图片大小
        im2_mask = get_face_mask(im2_size, landmarks2)  # 摄像头图片人脸掩模

        affine_im1 = get_affine_image(im1, im2, landmarks1, landmarks2)  # im1（脸图）仿射变换后的图片
        affine_im1_mask = get_affine_image(im1_mask, im2, landmarks1, landmarks2)  # im1（脸图）仿射变换后的图片的人脸掩模

        union_mask = get_mask_union(im2_mask, affine_im1_mask)  # 掩模合并


        affine_im1 = skin_color_adjustment(affine_im1, im2, mask=union_mask)  # 肤色调整
        point = get_mask_center_point(affine_im1_mask)  # im1（脸图）仿射变换后的图片的人脸掩模的中心点
        seamless_im = cv2.seamlessClone(affine_im1, im2, mask=union_mask, p=point, flags=cv2.NORMAL_CLONE)  # 进行泊松融合
        
        new_im = os.path.join(new_path, os.path.split(im_name2)[-1])
        cv2.imwrite(new_im, seamless_im)
    else:
        shutil.copy(im_name2, new_path)


# 按帧提取图片
def cut_video_to_image(video_path, img_path):
    """
    将视频分解为图片
    输入参数: 分割视频地址,保存图片地址
    输出参数: 声明了两个全局变量，用来保存分割图片的大小
    功能：将视频按帧分割成图片
    """
    cap = cv2.VideoCapture(video_path)

    index = 0
    global size_y, size_x, fps

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    while(True):
        ret,frame = cap.read() 
        if ret:
            cv2.imwrite(img_path + '/%d.jpg' % index, frame)
            index += 1
        else:
            break
        size_x = frame.shape[0]
        size_y = frame.shape[1]
    cap.release()

    print('Video cut finish, all %d frame' % index)
    print("imge size:x is {1},y is {0}".format(size_x,size_y))


# 图片换脸
def get_faces_changed(im_name1, old_path, new_path):
    """
    循环读取图片，并进行换脸
    :param im_name1: 要替换成的人脸图片文件名
    :param old_path: 原始图片目录
    :param new_path: 换脸后输出图片目录
    """
    im1 = cv2.imread(im_name1)
    im_names = os.listdir(old_path)
    for im_name2 in os.listdir(old_path):
        if im_name2.startswith("."):
            continue
        
        change_face(im1, old_path+"/"+im_name2, new_path)

# 提取原音频
def combine_audio_video(orig_video, new_video_wot_audio, final_video):
    """
    提取原视频中的音频，并与修改后的视频融合
    """
    audioclip = AudioFileClip(orig_video)
    clip_finall = VideoFileClip(new_video_wot_audio)
    videoclip = clip_finall.set_audio(audioclip)
    videoclip.write_videofile(final_video)

# 将换好的图片 转换成视频
def combine_image_to_video(image_path, video_name):
    """
    视频合成

    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #print(video_name)
    out = cv2.VideoWriter(video_name, fourcc, fps, (size_y, size_x))
    files = os.listdir(image_path)
    print("一共有{}帧图片要合成".format(len(files)))
    for i in range(len(files)):
        pic_name = str(i)+".jpg"
        #print(pic_name)
        img = cv2.imread(image_path + "/" + pic_name)
        out.write(img)    # 保存帧
    out.release()

# 主体函数
def main(orig_video, face_img, out_video, video_to_img_path="frame", face_changed_img_path="frame_changed"):
   
    ## 视频切割为图片
    print("将视频切割为图片", end="\n\n")
    cut_video_to_image(orig_video, video_to_img_path)
    ## 换脸
    print("开始换脸", end="\n\n")
    get_faces_changed(face_img, video_to_img_path, face_changed_img_path)
    ## 将图片合成视频
    print("将图片合成视频", end="\n\n")
    combine_image_to_video(face_changed_img_path,"temp.mp4")
    ## 添加音频
    print("添加音频", end="\n\n")
    combine_audio_video(orig_video, "temp.mp4", out_video)
    print("Done!")


# 实例化对象
## 人脸关键点模型
face_landmark = hub.Module(name="face_landmark_localization")
## 视频人脸替换
main("video/jt_s.mp4", "xieshanshan_2020300829.jpg", "video/js_xieshanshan.mp4")


