import cv2
import numpy as np
from extraction.Fp import getOriGrad, enhance_fingerprint
from compute_freq import compute_global_freq
from pathlib import Path


def drawOrientation(ori, background, mask=None, block_size=16, color=(255, 0, 0), thickness=1, is_block_ori=False):
    '''
    :param im: the background image to draw orientation field (in place, backup outside)
    :param ori: the pixel-wise orientation field
    :param blks_zie: the block size of orientation showing
    :param color: the line color showing the orientation
    :param thickness: the line thickness
    :return: im
    '''
    h = ori.shape[0]
    w = ori.shape[1]
    if is_block_ori:
        draw_step = 1
        offset = 0
    else:
        draw_step = block_size
        offset = int(block_size / 2)
    for x in range(0, w - offset, draw_step):
        for y in range(0, h - offset, draw_step):
            if mask is not None and mask[y + offset, x + offset] == 0:
                continue
            th = ori[y + offset, x + offset]
            if is_block_ori:
                x0 = x * block_size + block_size / 2
                y0 = y * block_size + block_size / 2
            else:
                x0 = x + block_size / 2
                y0 = y + block_size / 2
            x1 = int(x0 + 0.4 * block_size * np.cos(th))
            y1 = int(y0 + 0.4 * block_size * np.sin(th))
            x2 = int(x0 - 0.4 * block_size * np.cos(th))
            y2 = int(y0 - 0.4 * block_size * np.sin(th))
            cv2.line(background, (x1, y1), (x2, y2), color, thickness)
    return background

def showOrientation(ori, background=None, mask=None, block_size=16, color=(0,0,255), thickness=1,
                    win_name='orientation', wait_to_show=False):
    '''
    show orientation field in block wise. the orientation in each block is showing with a short line
    :param ori: the input orientation field in pixel-wise
    :param background: the background image to draw the lines. can be None, then show on a white image
    :param mask: the pixel-wise mask image
    :param block_size: the block size to draw orientation
    :param color: the color of line
    :param thickness: line thickness
    :param win_name: the window name
    :return: None
    '''
    if background is None:
        bg = np.ones((ori.shape[0], ori.shape[1], 3), dtype=np.uint8) * 255
    elif len(background.shape)==2:
        bg = np.stack((background, background, background), 2)
    else:
        bg = background.copy()

    im_show = drawOrientation(ori, bg, mask=mask, block_size=block_size, color=color, thickness=thickness)
    cv2.imshow(win_name, im_show)
    if not wait_to_show:
        cv2.waitKey()


if __name__ == '__main__':
    store_path = Path('/media/nutalk/data/collection/store_1126_dry/export/aict')

    enroll_path = Path('/media/nutalk/data/collection/store_1126_dry/export/aict/users/0022/enroll/')
    verify_path = Path('/media/nutalk/data/collection/store_1126_dry/export/aict/users/0022/verify/')
    
    output_path = Path('/media/nutalk/data/collection/store_1126_dry/export/gabor')
    
    f_rec = []
    for enroll_img_path in enroll_path.glob("*.png"):
        img_path = str(enroll_img_path)
        image = cv2.imread(img_path, 0)
        ori = getOriGrad(image, w=31)
        h, w = image.shape
        h1, w1 = int(h / 2), int(w / 2)
        f = 1. / compute_global_freq(image[h1-41:h1+41,w1-41:w1+41])
        f_rec.append(f)
        image_enhance = enhance_fingerprint(image, ori, f=f, band_width=4.5)

        image_out_path = output_path / enroll_img_path.relative_to(store_path)
        image_out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_out_path),image_enhance)

    f_average = np.average(f_rec)
        
    for verify_img_path in verify_path.glob('*.png'):
        image = cv2.imread(str(verify_img_path), 0)
        ori = getOriGrad(image, w=31)
        h, w = image.shape
        h1, w1 = int(h / 2), int(w / 2)
        image_enhance = enhance_fingerprint(image, ori, f=f_average, band_width=4.5)

        image_out_path = output_path / verify_img_path.relative_to(store_path)
        image_out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_out_path), image_enhance)
