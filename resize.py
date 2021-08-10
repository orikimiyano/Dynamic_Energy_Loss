import os
import os.path
from PIL import Image
import cv2


def ResizeImage(filein, fileout, width, height, type) :
    img = Image.open(filein)
    out = img.resize((width, height), Image.ANTIALIAS)
    # resize image with high-quality
    out.save(fileout, type)


if __name__ == "__main__" :
    path = "./data_1/gt/Case1"  # 待读取的文件夹
    path_list = os.listdir(path)
    path_list.sort()  # 对读取的路径进行排序
    for filename in path_list :

        filein = path+'/'+filename
        fileout = path+'/'+filename
        type = 'png'
        img = cv2.imread(path+'/'+filename)
        print (path+'/'+filename)

        width = 256
        height = 256

        ResizeImage(filein, fileout, width, height, type)
