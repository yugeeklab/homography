import numpy as np
from PIL import Image
'''
occlusion filling을 수행한다.
간단하게 이전 column에 있는 값을 가져온다.
'''
def occlusion_filling(img) :
    row_size, col_size = img.shape

    for row in range(row_size) :
        for col in range(col_size) :
            if(img[row][col] == 0 and col > 0) :
                img[row][col] = img[row][col-1]

    return img

if __name__ == '__main__' :
    img = Image.open('norm_version_new3.png').convert('L')
    img = np.asarray(img)

    occlusion_filling(img)

    Image.fromarray(img).save('after_occlusion.png')