import os

import numpy as np
import cv2
import numba

import matplotlib.pyplot as plt


def energy(img_gray):
    img_float = img_gray.astype(np.float)
    Ix, Iy = np.gradient(img_float)
    e = np.abs(Ix) + np.abs(Iy)
    return e


def cum_energy_vertical(e):
    n_rows, n_cols = e.shape
    M = np.empty(shape=(n_rows, n_cols + 2))

    # pad with infinity
    M[:, [0, -1]] = np.inf
    M[:, 1:-1] = e

    # dynamic programming
    M = cum_energy_vertical_loop(M)

    return M


@numba.jit
def cum_energy_vertical_loop(M):
    n, m = M.shape
    for i in range(1, n):
        for j in range(1, m - 1):
            M[i, j] = M[i, j] + min(M[i - 1, j - 1], M[i - 1, j], M[i - 1, j + 1])
    return M


def fwd_energy(img_gray):
    Cl = np.empty_like(img_gray, dtype=np.float)
    Cu = np.empty_like(img_gray, dtype=np.float)
    Cr = np.empty_like(img_gray, dtype=np.float)

    img_gray = img_gray.astype(np.float)

    fwd_energy_loop(Cl, Cr, Cu, img_gray)

    return Cl, Cu, Cr


@numba.jit
def fwd_energy_loop(Cl, Cr, Cu, img_gray):
    n, m = img_gray.shape
    for i in range(n):
        for j in range(m):
            j_l = j - 1 if j > 0 else j
            j_r = j + 1 if j < m - 1 else j

            row_diff = np.abs(img_gray[i, j_r] - img_gray[i, j_l])

            if i == 0:
                Cl[i, j] = np.inf
                Cu[i, j] = row_diff
                Cr[i, j] = np.inf
            else:
                Cl[i, j] = row_diff + np.abs(img_gray[i - 1, j] - img_gray[i, j_l])
                Cu[i, j] = row_diff
                Cr[i, j] = row_diff + np.abs(img_gray[i - 1, j] - img_gray[i, j_r])


def fwd_cum_energy_vertical(e, P=None):
    n_rows, n_cols = e[0].shape
    M = np.empty(shape=(n_rows, n_cols + 2))

    # pad with infinity
    M[:, [0, -1]] = np.inf
    M[:, 1:-1] = np.zeros_like(M[:, 1:-1])

    # dynamic programming
    if P is None:
        P = np.zeros((n_rows, n_cols))

    M = fwd_cum_energy_vertical_loop(M, e, P)

    return M


@numba.jit
def fwd_cum_energy_vertical_loop(M, e, P):
    Cl, Cu, Cr = e
    n, m = M.shape
    for i in range(1, n):
        for j in range(1, m - 1):
            M[i, j] = P[i, j] + min(
                M[i - 1, j - 1] + Cl[i, j],
                M[i - 1, j] + Cu[i, j],
                M[i - 1, j + 1] + Cr[i, j],
            )

    return M


def find_seam_vertical(M):
    mask = np.full_like(M, fill_value=False, dtype=np.bool)

    mask = find_seam_vertical_loop(M, mask)

    return mask[:, 1:-1]


@numba.jit
def find_seam_vertical_loop(M, mask):
    n, m = M.shape
    j = np.argmin(M[n - 1, :])
    mask[n - 1, j] = True
    for i in range(n - 2, -1, -1):
        offset = np.argmin(M[i, j - 1:j + 2])
        j = j - 1 + offset
        mask[i, j] = True
    return mask


def shrink_vertically(image, new_width, is_fwd=False, P=None):
    m_new = new_width if new_width > 1 else int(new_width * image)

    img_resized_bgr = np.copy(image)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    n, m = img_gray.shape

    while m > m_new:
        e = fwd_energy(img_gray) if is_fwd else energy(img_gray)
        M = fwd_cum_energy_vertical(e, P) if is_fwd else cum_energy_vertical(e)

        # plt.imshow(M[1:-1])
        # plt.show()

        s = find_seam_vertical(M)

        # remove seam in grayscale image
        img_gray = np.reshape(img_gray[~s], (n, m - 1))

        # remove seam in BGR image
        img_resized_bgr = np.reshape(img_resized_bgr[~s], (n, m - 1, 3))

        # update shape
        n, m = img_gray.shape

    return img_resized_bgr


def allign_seams(img_gray, seams):
    n, m = img_gray.shape

    mask = np.full_like(img_gray, fill_value=False, dtype=np.bool)

    removed_columns = {row_i: [] for row_i in range(n)}

    for s_i, s in enumerate(seams):
        for row_i in range(n):
            j = np.argwhere(s[row_i] == True).flatten()[0]

            previously_removed = removed_columns[row_i]
            n_removed = len([x for x in previously_removed if x <= j])

            j_adj = j + n_removed

            while mask[row_i, j_adj] == True:
                j_adj += 1

            removed_columns[row_i].append(j_adj)
            mask[row_i, j_adj] = True

    k = len(seams)

    # print(np.sum(mask))
    # print(np.sum(mask) / k)
    # print(k * n)

    assert np.sum(mask) == k * n

    return mask


def draw_seams(image, k, is_fwd=False, P=None):
    img_bgr = np.copy(image).astype(np.float)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seams = find_seams_vertical(img_gray, k, is_fwd, P)
    mask = allign_seams(img_gray, seams)

    img_bgr[:, :, 0][mask] = 0
    img_bgr[:, :, 1][mask] = 0
    img_bgr[:, :, 2][mask] = 255

    return img_bgr


def expand_img_with_mask(img_bgr, mask, k):
    n, m, ch = img_bgr.shape

    img_bgr_exp = np.full(shape=(n, m + k, ch), fill_value=0., dtype=np.float)
    img_bgr_seams = np.full(shape=(n, m + k, ch), fill_value=0., dtype=np.float)

    for c in range(ch):
        col_adj = {i: 0 for i in range(n)}

        for i in range(n):
            for j in range(m):
                flag = mask[i, j]

                j_exp = j + col_adj[i]

                if flag:  # adds 2 values to the canvas
                    if j == 0:
                        # image borders
                        left_v = img_bgr[i, j, c]
                        right_v = (img_bgr[i, j, c] + img_bgr[i, j + 1, c]) / 2.
                        img_bgr_exp[i, j_exp, c] = left_v
                        img_bgr_exp[i, j_exp + 1, c] = right_v
                    elif j == m - 1:
                        left_v = (img_bgr[i, j, c] + img_bgr[i, j - 1, c]) / 2.
                        right_v = img_bgr[i, j, c]
                        img_bgr_exp[i, j_exp, c] = left_v
                        img_bgr_exp[i, j_exp + 1, c] = right_v
                    else:
                        # expanded image
                        left_v = (img_bgr[i, j, c] + img_bgr[i, j - 1, c]) / 2.
                        right_v = (img_bgr[i, j, c] + img_bgr[i, j + 1, c]) / 2.
                        img_bgr_exp[i, j_exp, c] = left_v
                        img_bgr_exp[i, j_exp + 1, c] = right_v

                    # seams vizualization
                    img_bgr_seams[i, j_exp, c] = img_bgr[i, j, c]
                    img_bgr_seams[i, j_exp + 1, c] = 255 if c == 2 else 0

                    col_adj[i] += 1
                else:
                    img_bgr_exp[i, j_exp, c] = img_bgr[i, j, c]
                    img_bgr_seams[i, j_exp, c] = img_bgr[i, j, c]

    return img_bgr_exp, img_bgr_seams


def expand_vertically(image, p=0.5, is_fwd=False):
    if p < 1:
        k = int(p * image.shape[1])
    else:
        k = int(p - image.shape[1])

    img_bgr = np.copy(image).astype(np.float)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    seams = find_seams_vertical(img_gray, k, is_fwd)

    mask = allign_seams(img_gray, seams)

    img_bgr_exp, img_bgr_seams = expand_img_with_mask(img_bgr, mask, k)

    return img_bgr_exp, img_bgr_seams


def find_seams_vertical(img_gray, k, is_fwd, P=None):
    seams = []
    #
    # for i in range(n):
    #     mask = np.full_like(M, fill_value=False, dtype=np.bool)
    #
    #     mask = find_seam_vertical_loop(M, mask)
    #     seams.append(mask[:, 1:-1])

    # M[mask] = np.inf
    # M[-1, :][mask[-1, :]] = np.inf
    # M = cum_energy_vertical_loop(M)

    n, m = img_gray.shape

    for i in range(k):
        e = fwd_energy(img_gray) if is_fwd else energy(img_gray)
        M = fwd_cum_energy_vertical(e, P) if is_fwd else cum_energy_vertical(e)
        s = find_seam_vertical(M)

        seams.append(s)

        # remove seam in grayscale image
        img_gray = np.reshape(img_gray[~s], (n, m - i - 1))

        # remove seam in bgr image
        # img_bgr = np.reshape(img_bgr[~s], (n, m - i - 1, 3))

    return seams


# --------------------------------------------------------------- #
#                              RUNNER                             #
# --------------------------------------------------------------- #


if __name__ == '__main__':
    # REPLICATE FIGURE 5 (2007)
    fig5_path = os.path.join('input', 'fig5.png')
    im = cv2.imread(fig5_path)

    im_resized = shrink_vertically(im, 350)

    cv2.imwrite(os.path.join('output', 'fig5.png'), im_resized)

    # REPLICATE FIGURE 8 (2007)
    fig8_path = os.path.join('input', 'fig8.png')
    img = cv2.imread(fig8_path)

    img_bgr_exp, img_bgr_seams = expand_vertically(img, 0.5)

    cv2.imwrite(os.path.join('output', 'fig8c.png'), img_bgr_seams)
    cv2.imwrite(os.path.join('output', 'fig8d.png'), img_bgr_exp)

    img_bgr_exp_norm = np.zeros_like(img_bgr_exp)
    img_bgr_exp_norm = cv2.normalize(img_bgr_exp, img_bgr_exp_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    img_bgr_exp2, _ = expand_vertically(img_bgr_exp_norm, 479)
    cv2.imwrite(os.path.join('output', 'fig8e.png'), img_bgr_exp2)

    # REPLICATE FIGURE 8 (2008)
    fig8_2008_path = os.path.join('input', 'fig8-2008.png')
    im = cv2.imread(fig8_2008_path)

    im_resized = shrink_vertically(im, 256, is_fwd=False)
    cv2.imwrite(os.path.join('output', 'fig8-2008-a.png'), im_resized)

    im_resized_seams = draw_seams(im, k=im.shape[1] - 256, is_fwd=False)
    cv2.imwrite(os.path.join('output', 'fig8-2008-b.png'), im_resized_seams)

    # info about the image (~face detector)
    Pij = cv2.imread(os.path.join('input', 'fig8-2008', 'fig8-2008-Pij.png'))
    Pij = cv2.cvtColor(Pij, cv2.COLOR_BGR2GRAY)
    Pij[np.where(Pij > 20)] = 0
    Pij *= 5

    im_resized = shrink_vertically(im, 256, is_fwd=True, P=Pij)
    cv2.imwrite(os.path.join('output', 'fig8-2008-c.png'), im_resized)

    im_resized_seams = draw_seams(im, k=im.shape[1] - 256, is_fwd=True, P=Pij)
    cv2.imwrite(os.path.join('output', 'fig8-2008-d.png'), im_resized_seams)

    # REPLICATE FIGURE 9 (2008)
    fig9_path = os.path.join('input', 'fig9-2008.png')
    img = cv2.imread(fig9_path)

    img_bgr_exp, _ = expand_vertically(img, 576, is_fwd=False)
    cv2.imwrite(os.path.join('output', 'fig9-2008-a.png'), img_bgr_exp)

    img_bgr_exp, _ = expand_vertically(img, 576, is_fwd=True)
    cv2.imwrite(os.path.join('output', 'fig9-2008-b.png'), img_bgr_exp)
