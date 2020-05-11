import os

import numpy as np
import cv2

from seam_carving import energy


def analyze(im, im_gt):
    diff = np.power(im.astype(np.float) - im_gt.astype(np.float), 2)
    sse = np.sum(diff)
    print(f'SSE with replica: {sse}')

    rand_image = np.random.randint(0, 255, size=(im_gt.shape))
    diff = np.power(rand_image.astype(np.float) - im_gt.astype(np.float), 2)
    sse = np.sum(diff)
    print(f'SSE with random: {sse}')

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gt_gray = cv2.cvtColor(im_gt, cv2.COLOR_BGR2GRAY)
    e_im = np.sum(energy(im_gray))
    e_im_gt = np.sum(energy(im_gt_gray))
    print(f'Energy (replica): {e_im}')
    print(f'Energy (base): {e_im_gt}')


# --------------------------------------------------------------- #
#                              RUNNER                             #
# --------------------------------------------------------------- #


if __name__ == '__main__':
    np.random.seed(42)

    # REPLICATE FIGURE 5 (2007)
    print('\nCoastal')

    fig_path = os.path.join('output', 'fig5.png')
    fig_gt_path = os.path.join('input', 'fig5', 'waterfallNarrow.png')
    im = cv2.imread(fig_path)
    im_gt = cv2.imread(fig_gt_path)

    analyze(im, im_gt)


    # REPLICATE FIGURE 8 (2007)
    print('\nDolphin 50')
    fig_path = os.path.join('output', 'fig8d.png')
    fig_gt_path = os.path.join('input', 'fig8', 'dolphinStretch1.jpg')
    im = cv2.imread(fig_path)
    im_gt = cv2.imread(fig_gt_path)

    analyze(im, im_gt)

    print('\nDolphin 100')

    fig_path = os.path.join('output', 'fig8e.png')
    fig_gt_path = os.path.join('input', 'fig8', 'dolphinStretch2.jpg')
    im = cv2.imread(fig_path)
    im_gt = cv2.imread(fig_gt_path)

    analyze(im, im_gt)


    # REPLICATE FIGURE 8 (2009)
    print('\nBench BWD')
    fig_path = os.path.join('output', 'fig8-2008-a.png')
    fig_gt_path = os.path.join('input', 'fig8-2008', 'bench3_bwd.png')
    im = cv2.imread(fig_path)
    im_gt = cv2.imread(fig_gt_path)

    analyze(im, im_gt)

    print('\nBench FWD')

    fig_path = os.path.join('output', 'fig8-2008-c.png')
    fig_gt_path = os.path.join('input', 'fig8-2008', 'bench3_fwd.png')
    im = cv2.imread(fig_path)
    im_gt = cv2.imread(fig_gt_path)

    analyze(im, im_gt)

    # REPLICATE FIGURE 9 (2009)
    print('\nCar BWD')
    fig_path = os.path.join('output', 'fig9-2008-a.png')
    fig_gt_path = os.path.join('input', 'fig9-2008', 'bwd.png')
    im = cv2.imread(fig_path)
    im_gt = cv2.imread(fig_gt_path)

    analyze(im, im_gt)

    print('\nCar FWD')

    fig_path = os.path.join('output', 'fig9-2008-b.png')
    fig_gt_path = os.path.join('input', 'fig9-2008', 'fwd.png')
    im = cv2.imread(fig_path)
    im_gt = cv2.imread(fig_gt_path)

    analyze(im, im_gt)