import numpy as np
from PIL import Image

def do_norm(img) :
    img = (img / img.max()) * 255

    return img

if __name__ == '__main__' :
    img = Image.open('30-60-100-full/full.png').convert('L')
    img = np.asarray(img).astype(np.uint8)

    do_norm(img)

    Image.fromarray(img).save('after_norm.png')