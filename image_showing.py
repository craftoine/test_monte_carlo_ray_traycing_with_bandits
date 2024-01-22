"""images are txt files shuch as:
first line is 3
second is shape[0] shape[1] 3
then ther is shape[0] lises of lentth shape[1] * 3
"""
import numpy as np
import matplotlib.pyplot as plt


def import_image(path):
    """import image from path"""
    with open(path, 'r') as f:
        lines = f.readlines()
    try:
        print('try image as int')
        """try image as int"""
        shape = [int(i) for i in lines[1].split()]
        image = np.zeros(shape)
        for i in range(shape[0]):
            ar = list(map(int,lines[i + 2].split()))
            for j in range(shape[1]):
                for k in range(3):
                    image[i][j][k] = ar[j * 3 + k]
        return image,True
    except:
        print('try image as float')
        """try image as float"""
        shape = [int(i) for i in lines[1].split()]
        image = np.zeros(shape+[3])
        for i in range(shape[0]):
            ar = list(map(float,lines[i + 2].split()))
            for j in range(shape[1]):
                image[i][j] = ar[j * 3: j * 3 + 3]
        return image,False
def show_image(image):
    """show image"""
    plt.imshow(image)
    plt.show()

while True:
    try:
        path = input('enter path: ')
        image,b = import_image(path)
    except:
        print('wrong path')
    if b:
        show_image(image / 255)
    else:
        show_image(image)