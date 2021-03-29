#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# In[2]:


if __name__ == '__main__':

    f = open("test.txt", "w+")
    try:

        def getallfile(inputpath,outputpath):
            allfilelist = os.listdir(inputpath)
            for file in allfilelist:
                filepath = os.path.join(inputpath, file)
                # 判断是不是文件夹
                if os.path.isdir(filepath):
                    getallfile(filepath, os.path.join(outputpath, file))
                else:
                    if (filepath.lower().endswith(
                            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
                        #print(filepath)
                        f.write(filepath + ' ' + filepath + '\n')



        getallfile("input", "output")
    finally:
        f.close()







