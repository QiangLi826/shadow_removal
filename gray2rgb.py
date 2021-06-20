# gray2rgb&resize This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import cv2
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

#加通道缩小1/4
def convertImg(path, num):
    print("happy...")
    for root, dirs, files in os.walk(path):
        for name in files:
            if (name.endswith('.jpg')):
                path_img = os.path.join(root, name)

                print("path_img:    " + path_img)

                img1 = cv2.imdecode(np.fromfile(path_img,dtype=np.uint8), 0)
                print(img1)
                print(img1.shape)
                height, width = img1.shape[:2]
                size = (int(width * 0.5), int(height * 0.5))
                img1 = cv2.resize(img1, size, interpolation=cv2.INTER_AREA)
                print(img1.shape)
                #img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
                # cv2.imshow('img1_rgb', img1_rgb)
                print(num + name)
                cv2.imwrite(num + name, img1)
                # name='1/'+name
                # txt.write(name+" "+name+'\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    path = r'D:\work\道路图片\第四条路'
    # txtName='1.txt'
    # txt = open(txtName, 'w')
    convertImg(path,r'D:\work\4road_scale_025\\')
    # txt.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



