#导入所需的库
import os
import zipfile
import cv2
import math
import numpy as np
from PIL import Image
import paddlehub as hub
from moviepy.editor import *

# 1、将视频切割成图片

bg_music = '/home/aistudio/movies/music.mp4'    #背景音乐
video_people = '/home/aistudio/movies/people.mp4' #人物视频
video_bakgroud = '/home/aistudio/movies/bg.mp4'    #背景视频

people_path = "/home/aistudio/frames/people/"     #视频转换后人物目录
people_newpath = "/home/aistudio/frames/people1/people/" #处理后可以抠图的人物目录
backgroud_path = "/home/aistudio/frames/bg/"        #视频转换后背景目录

path = '/home/aistudio/output/'   #人物抠图后输出路径
middle_video='/home/aistudio/movies/green.mp4'
output_video = '/home/aistudio/movies/result.mp4'


def getFrame(video_name, save_path):#将视频逐帧保存为图片

    video = cv2.VideoCapture(video_name)
    # 获取视频帧率
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    # 获取画面大小
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    # 获取帧数
    frame_num = str(video.get(7))
    name = int(math.pow(10, len(frame_num)))
    ret, frame = video.read()
    while ret:
        cv2.imwrite(save_path + str(name) + '.jpg', frame)
        ret, frame = video.read()
        name += 1
    video.release()
    return fps, size, frame_num

def getFrame1(video_name, save_path):#将视频逐帧保存为图片
    video = cv2.VideoCapture(video_name)
    # 获取视频帧率
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    # 获取画面大小
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    # 获取帧数
    frame_num = str(video.get(7))
    return fps, size, frame_num



#将视频按帧保存为图片
#人物视频转换图片
if not os.path.exists(people_path):
    os.makedirs(people_path)
fps, size, frame_number = getFrame(video_people, people_path)

#背景视频转换图片
if not os.path.exists(backgroud_path):
    os.makedirs(backgroud_path)
fps2, size2, frame_number2 = getFrame(video_bakgroud, backgroud_path)



# 2、由于视频质量问题，需要手动清除掉切图后的部分无效图片

def unzip_data():
    '''
    解压原始数据集，将src_path路径下的zip包解压至target_path目录下
    '''
    if(not os.path.isdir('/home/aistudio/frames/people1/')):     
        z = zipfile.ZipFile('/home/aistudio/work/people.zip', 'r')
        z.extractall(path='/home/aistudio/frames/people1/')
        z.close()
unzip_data()

# 3、对有效图，进行人物抠图操作

def getHumanseg(frames):#对帧图片进行批量抠图
    humanseg = hub.Module(name='deeplabv3p_xception65_humanseg')
    files = [frames + i for i in os.listdir(frames)]
    humanseg.segmentation(data={'image': files}, visualization=True, use_gpu=True,
                                    output_dir=path) 


#使用GPU进行抠图
%set_env CUDA_VISIBLE_DEVICES=0


# 抠好的图片位置
if not os.path.exists(path):
    os.makedirs(path)
#批量抠图
getHumanseg(people_newpath)


4、将人物和背景组合视频


fps, size, frame_number = getFrame1(video_people, people_path)

def readBg(src,size):#读取背景图片，修改尺
    im = Image.open(src)
    return im.resize(size)

def setImageBg(humanseg, bg_im):#将抠好的图和背景图片合并
    # 读取抠完后的图片
    im = Image.open(humanseg)
    # 分离色道
    r, g, b, a = im.split()
    bg_im = bg_im.copy()
    bg_im.paste(im, (0, 0), mask=a)
    return np.array(bg_im.convert('RGB'))[:, :, ::-1]

def writeVideo(path, bg_im, fps, size):# 写入视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(middle_video, fourcc, fps, size)
    # 为每一帧设置背景
    files = os.listdir(path)
    #将文件名按编号顺序排序
    file_num_sort = []
    for file in files:
        if file.endswith('.png'):
            file_num_sort.append(int(file.split('.')[0]))
    file_num_sort.sort()
    file_sort=[]
    for file_num in file_num_sort:
        file_sort.append(path+str(file_num)+'.png')# 真实用
    for file_num in file_num_sort:
        file_sort.append(path+str(file_num)+'.png')# 真实用
    i=0
    for file in file_sort:
        im_array = setImageBg(file, bg_im[i])
        if i<len(bg_im)-1:
            i=i+1
        out.write(im_array)
    out.release()


# 最终视频的保存路径
bg_im=[]
img_paths = os.listdir(backgroud_path)
img_paths.sort()
for img in img_paths:
    bg_im.append(readBg(backgroud_path +img,size)) #防止重影
writeVideo(path, bg_im, fps, size)





