import os

import cv2
import numpy as np
import scipy
from PIL.ImageDraw import ImageDraw
from lmfit import Parameters, minimize, report_fit

from matplotlib import pyplot as plt
from PIL import Image

import matplotlib.colors as mcolors

from pca import RobustPCA
from timestamp_splitter_starlight import load_counts
from astropy.io import fits

from util import show_image

plt.style.use('seaborn-v0_8-deep')

def percentile_stretching(img, cutoff_percentile: int = 1):
    # Removes 99th percentile of noise
    min_val = np.percentile(img, cutoff_percentile)
    mean_val = np.percentile(img, 100-cutoff_percentile)

    return np.clip(((img-min_val)/(mean_val-min_val)*255), a_min=0, a_max=255)

def load_folder(folder_path: str, show_image: bool = False, maskx=None, masky=None, dark=None):
    img_stack = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".fit"):
            print(f"{filename} not a fit file")
            continue

        path = folder_path + "\\" + filename
        try:
            img = fits.open(path)[0].data

        except FileNotFoundError:
            print(f"{path} not a fits file")
            return None

        if masky is not None:
            img = img[masky[0]:masky[1], :]

        if maskx is not None:
            img = img[:, maskx[0]:maskx[1]]

        if show_image:
            preview_img = percentile_stretching(img, 10)
            plt.title(f"{filename}")
            plt.imshow(preview_img)
            plt.show()

        img_stack.append(img)

    return np.median(img_stack, axis=0)

def flat_field_correction(img: np.ndarray, bg: np.ndarray, dark: np.ndarray | None = None):
    if dark is None:
        dark = np.zeros_like(img)

    gain = np.mean(bg) / (bg - dark)
    corr = (img - dark) * gain

    return corr

def create_huge_image(images, ncols, nrows, names):
    if ncols * nrows < len(names):
        raise ValueError("ncols * nrows must be equal or greater than number of images!")

    single_width, single_height = images[0].size
    print(single_width, single_height)

    full_height = single_height * nrows
    full_width = single_width * ncols

    new_im = Image.new('RGB', (full_width, full_height))
    new_im_labelled = Image.new('RGB', (full_width, full_height))

    draw = ImageDraw(new_im_labelled)

    img_index = 0
    for j in range(nrows):
        for i in range(ncols):
            if img_index >= len(names):
                continue

            img = images[img_index]
            #img.save(f"{save_dest}/{names[img_index]}.png")

            pos = (i*single_width, j*single_height)
            new_im.paste(img, pos)
            new_im_labelled.paste(img, pos)

            img_index += 1

    return new_im_labelled

def run_pca(median, show_image=True):
    rpca = RobustPCA(median)
    L, S = rpca.fit(max_iter=200, iter_print=20)

    if show_image:
        """
        plt.figure()
        plt.title(f"Median image at {folder_path}")
        plt.imshow(median)
        plt.show()
        """

        if show_image:
            fig, ax = plt.subplots(ncols=3)
            ax[0].imshow(L)
            ax[0].set_title("low rank data")
            ax[1].imshow(S)
            ax[1].set_title("sparse data")
            ax[2].imshow(median)
            ax[2].set_title("original")
            plt.show()

    return median, L, S

def percentile_stretching_2(img, perc, overflow):
    mean_val = np.percentile(img, 100 - perc)
    img[img > mean_val] = overflow

    return img

def fit_2d_gaussian(img,intensity, offset, ypos=None, xpos=None, plot=False, vary_centroid: bool=True):
    def residuals(p, x, y, z):
        height = p["height"].value
        cen_x = p["centroid_x"].value
        cen_y = p["centroid_y"].value
        sigma_x = p["sigma_x"].value
        sigma_y = p["sigma_y"].value
        offset = p["background"].value

        f = gaussian2D(x, y, cen_x, cen_y, sigma_x, sigma_y, offset, height)
        return z - f

    xf = np.arange(0, img.shape[0])
    yf = np.arange(0, img.shape[1])

    xf, yf = np.meshgrid(yf, xf)

    if ypos is None:
        ypos = img.shape[0] / 2

    if xpos is None:
        xpos = img.shape[1] / 2

    initial = Parameters()
    initial.add("height", value=intensity, min=0)
    initial.add("background", value=offset, vary=True)
    initial.add("centroid_x", value=xpos, vary=True)
    initial.add("centroid_y", value=ypos, vary=vary_centroid)
    initial.add("sigma_x", value=-200, max=-150)
    initial.add("sigma_y", value=400, min=200)

    init_curve = gaussian2D(xf, yf,
                                   initial["centroid_x"],
                                   initial["centroid_y"],
                                   initial["sigma_x"],
                                   initial["sigma_y"],
                                   initial["background"],
                                   initial["height"]).reshape((img.shape))

    if plot:
        fig, ax = plt.subplots(ncols=2, nrows=1)

        plt.axis("off")
        ax[0].set_title("Image")
        ax[0].imshow(img, cmap="magma")

        ax[1].set_title("Initial background")
        ax[1].imshow(init_curve, cmap="magma")

        plt.show()

    fit = minimize(residuals, initial, args=(xf, yf, img))
    print(report_fit(fit))

    res = fit.params

    fit_curve = gaussian2D(xf, yf,
                                  res["centroid_x"],
                                  res["centroid_y"],
                                  res["sigma_x"],
                                  res["sigma_y"],
                                  res["background"],
                                  res["height"]).reshape((img.shape))

    init_curve = gaussian2D(xf, yf,
                            initial["centroid_x"],
                            initial["centroid_y"],
                            initial["sigma_x"],
                            initial["sigma_y"],
                            initial["background"],
                            initial["height"]).reshape((img.shape))

    if plot:
        fig, ax = plt.subplots(ncols=3, nrows=1)

        plt.axis("off")
        ax[0].set_title("Image")
        ax[0].imshow(img, cmap="magma")

        ax[1].set_title("Initial background")
        ax[1].imshow(init_curve, cmap="magma")

        ax[2].set_title("Fitted background")
        ax[2].imshow(fit_curve, cmap="magma")
        plt.show()

    return fit_curve, res


def fit_2d_gaussian_masked(img, offset, intensity, maskx, masky, plot=False, xpos=None, ypos=None):
    masked_corr = img.copy()
    masked_corr[masky[0]:masky[1], maskx[0]:maskx[1]] = 0

    def gaussian2D(x, y, x0, y0, sx, sy, offset, ampl):
        return ampl * np.exp(-(x - x0) ** 2 / (2 * sx ** 2) - (y - y0) ** 2 / (2 * sy ** 2)) + offset

    def residuals(p, x, y, z):
        height = p["height"].value
        cen_x = p["centroid_x"].value
        cen_y = p["centroid_y"].value
        sigma_x = p["sigma_x"].value
        sigma_y = p["sigma_y"].value
        offset = p["background"].value

        f = gaussian2D(x, y, cen_x, cen_y, sigma_x, sigma_y, offset, height)
        f[masky[0]:masky[1], maskx[0]:maskx[1]] = 0
        return z - f

    xf = np.arange(0, masked_corr.shape[0])
    yf = np.arange(0, masked_corr.shape[1])

    xf, yf = np.meshgrid(yf, xf)

    if xpos is None:
        xpos = masked_corr.shape[1] / 2

    if ypos is None:
        ypos = 2* masked_corr.shape[0] / 3

    initial = Parameters()
    initial.add("height", value=intensity, min=0)
    initial.add("background", value=offset, vary=True)
    initial.add("centroid_x", value=xpos, vary=True)
    initial.add("centroid_y", value=ypos, vary=True)
    initial.add("sigma_x", value=masked_corr.shape[0] / 4)
    initial.add("sigma_y", value=masked_corr.shape[0] / 4)

    fit = minimize(residuals, initial, args=(xf, yf, masked_corr))
    print(report_fit(fit))

    res = fit.params

    fit_curve_raw = gaussian2D(xf, yf,
                               res["centroid_x"],
                               res["centroid_y"],
                               res["sigma_x"],
                               res["sigma_y"],
                               res["background"],
                               res["height"]).reshape((masked_corr.shape))

    init_curve = gaussian2D(xf, yf,
                               initial["centroid_x"],
                               initial["centroid_y"],
                               initial["sigma_x"],
                               initial["sigma_y"],
                               initial["background"],
                               initial["height"]).reshape((masked_corr.shape))

    fit_curve = fit_curve_raw.copy()

    fit_curve_raw[masky[0]:masky[1], maskx[0]:maskx[1]] = 0

    if plot:
        fig, ax = plt.subplots(ncols=3, nrows=2)
        plt.axis("off")
        ax[0][0].set_title("Raw image")
        ax[0][0].imshow(img, cmap="magma")

        ax[1][0].set_title("Masked image")
        ax[1][0].imshow(masked_corr, cmap="magma")

        ax[0][1].set_title("Initial background")
        ax[0][1].imshow(init_curve, cmap="magma")

        ax[1][1].set_title("Fitted background")
        ax[1][1].imshow(fit_curve, cmap="magma")

        ax[0][2].set_title("Corrected image")
        ax[0][2].imshow(img / fit_curve, cmap="magma")

        plt.show()

    return fit_curve


def open_beam_normalisation(img, bg, norm_scale, p1, p2, plot=False):
    img = img - np.min(img)
    bg = bg - np.min(bg)

    # open beam normalisation on surrounding frame
    x_min, y_min = p1
    x_max, y_max = p2

    mask_copy = np.copy(img)
    mask_copy[y_min:y_max, x_min:x_max] = 0

    n_img = np.sum(mask_copy)

    img = (img / n_img) * norm_scale

    mask_copy_bg = np.copy(bg)
    mask_copy_bg[y_min:y_max, x_min:x_max] = 0

    n_bg = np.sum(mask_copy_bg)
    bg = (bg / n_bg) * norm_scale

    if plot:
        plt.imshow(mask_copy, cmap="magma")
        plt.show()

    return img, bg

def gaussian2D(x, y, x0, y0, sx, sy, offset, ampl):
        return ampl * np.exp(-(x - x0) ** 2 / (2 * sx ** 2) - (y - y0) ** 2 / (2 * sy ** 2)) + offset

def skewed_gaussian2D(x, y, x0, y0, sx, sy, offset, ampl, skewx, skewy):
        dx = (x - x0) / sx
        dy = (y - y0) / sy

        g = np.exp(-0.5 * (dx ** 2 + dy ** 2))
        skew = scipy.stats.norm.cdf(skewx * dx + skewy * dy)
        return 2 * ampl * g * skew + offset

def fit_2d_gaussian_skewed(img,intensity, offset, skew_x=None, skew_y=None, ypos=None, xpos=None, plot=False):
    def residuals(p, x, y, z):
        height = p["height"].value
        cen_x = p["centroid_x"].value
        cen_y = p["centroid_y"].value
        sigma_x = p["sigma_x"].value
        sigma_y = p["sigma_y"].value
        offset = p["background"].value
        skew_x = p["skew_x"].value
        skew_y = p["skew_y"].value

        f = skewed_gaussian2D(x, y, cen_x, cen_y, sigma_x, sigma_y, offset, height, skew_x, skew_y)
        return z - f

    xf = np.arange(0, img.shape[0])
    yf = np.arange(0, img.shape[1])

    xf, yf = np.meshgrid(yf, xf)

    if ypos is None:
        ypos = img.shape[0] / 2

    if xpos is None:
        xpos = img.shape[1] / 2

    if skew_x is None:
        skew_x = 0

    if skew_y is None:
        skew_y = 0

    initial = Parameters()
    initial.add("height", value=intensity, min=0)
    initial.add("background", value=offset, vary=True)
    initial.add("centroid_x", value=xpos, vary=True)
    initial.add("centroid_y", value=ypos, vary=False)
    initial.add("sigma_x", value=-200, max=-150)
    initial.add("sigma_y", value=400, min=200)
    initial.add("skew_x", value=skew_x, vary=False)
    initial.add("skew_y", value=skew_y, min=-7)

    init_curve = skewed_gaussian2D(xf, yf,
                                   initial["centroid_x"],
                                   initial["centroid_y"],
                                   initial["sigma_x"],
                                   initial["sigma_y"],
                                   initial["background"],
                                   initial["height"],
                                   initial["skew_x"],
                                   initial["skew_y"]).reshape((img.shape))

    if plot:
        fig, ax = plt.subplots(ncols=2, nrows=1)

        plt.axis("off")
        ax[0].set_title("Image")
        ax[0].imshow(img, cmap="magma")

        ax[1].set_title("Initial background")
        ax[1].imshow(init_curve, cmap="magma")

        plt.show()

    fit = minimize(residuals, initial, args=(xf, yf, img))
    print(report_fit(fit))

    res = fit.params

    fit_curve = skewed_gaussian2D(xf, yf,
                                  res["centroid_x"],
                                  res["centroid_y"],
                                  res["sigma_x"],
                                  res["sigma_y"],
                                  res["background"],
                                  res["height"],
                                  res["skew_x"],
                                  res["skew_y"]).reshape((img.shape))

    init_curve = skewed_gaussian2D(xf, yf,
                            initial["centroid_x"],
                            initial["centroid_y"],
                            initial["sigma_x"],
                            initial["sigma_y"],
                            initial["background"],
                            initial["height"],
                            initial["skew_x"],
                            initial["skew_y"]).reshape((img.shape))

    if plot:
        fig, ax = plt.subplots(ncols=3, nrows=1)

        plt.axis("off")
        ax[0].set_title("Image")
        ax[0].imshow(img, cmap="magma")

        ax[1].set_title("Initial background")
        ax[1].imshow(init_curve, cmap="magma")

        ax[2].set_title("Fitted background")
        ax[2].imshow(fit_curve, cmap="magma")
        plt.show()

    return fit_curve, res

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def load_images(momentum_list: list[float],
                obj_path: str,
                dark_path: str,
                nexus_path: str,
                obj_outpath: str,
                maskx: tuple[int, int] | None = None,
                masky: tuple[int, int] | None = None,
                title: str="Imgs"):

    imgstack = []

    # Load number of counts in each image
    delayed_counts_obj = dict(sorted(load_counts(nexus_path).items()))

    # Load dark image
    dark_img = load_folder(dark_path, show_image=False, maskx=maskx, masky=masky)

    # Plot muon count distribution
    plt.figure()
    plt.title(title)
    plt.plot(delayed_counts_obj.keys(), delayed_counts_obj.values(), "o-", ms=5)
    plt.xlabel("Momentum")
    plt.ylabel("Delayed counts")
    plt.show()

    pixel_sums = []
    pixel_medians = []
    momenta = []

    # search for files in specified folder
    for momentum in momentum_list:
        obj_folder = obj_path + f"\\{momentum}"
        n_muons = delayed_counts_obj[momentum]

        try:
            # IMAGE
            obj_img = (load_folder(obj_folder, show_image=False, maskx=maskx, masky=masky))

        except FileNotFoundError:
            print(str(momentum) + f" data not found at {obj_folder}")
            continue

        print(f"Loaded folder {obj_folder}!")

        px_sum = np.sum(obj_img[0:300, :] + np.sum(obj_img[:, 0:100]))
        px_med = np.median(obj_img[0:300, :] + np.median(obj_img[:, 0:100]))

        pixel_medians.append(px_med)
        pixel_sums.append(px_sum)
        momenta.append(momentum)

        """
        n_target = np.sum(obj_img[:, 73]) + np.sum(obj_img[:, 831])
        n_dark = np.sum(dark_img[:, 73]) + np.sum(dark_img[:, 831])

        obj_img = (obj_img / n_target) * n_dark
        obj_img = obj_img - dark_img
        """

        n_dark = np.sum(obj_img[0:300, :] + np.sum(obj_img[:, 0:100]))

        obj_img = (obj_img / px_sum) * n_dark
        obj_img = obj_img - dark_img

        obj_img = percentile_stretching(obj_img, cutoff_percentile=1)
        #obj_img = scipy.ndimage.median_filter(obj_img, 3)

        pil_img = Image.fromarray(obj_img)
        pil_img.save(f"{obj_outpath}{momentum}_MED.tif")

        imgstack.append(pil_img)

    plt.figure()
    plt.title(title + "sum")
    plt.plot(momenta, pixel_sums)
    plt.xlabel("Momentum")
    plt.ylabel("Pixel sum")
    plt.show()

    plt.figure()
    plt.title(title + " med")
    plt.plot(momenta, pixel_medians)
    plt.xlabel("Momentum")
    plt.ylabel("Pixel medians")
    plt.show()

    return imgstack

if __name__ == '__main__':
    cmap = "magma"

    colors = ["pink", "purple", "black", "blue", "green"]
    nodes = [0.0, 0.3, 0.5, 0.7, 1]
    cmap2 = mcolors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    maskx_init = (100, -100)

    momentum_list_AAAA = [44, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74]
    momentum_list_AAAA_ff = momentum_list_AAAA

    momentum_list_9V = np.arange(56, 92, 2)
    momentum_list_9V_ff = np.arange(76, 92, 2)

    """
    imgs = load_images(momentum_list=momentum_list_AAAA,
                obj_path="D:\\raw_images\\AAAA",
                dark_path="D:\\raw_images\\AAAA\\46",
                nexus_path="C:\\Users\\sec90771\\OneDrive - University College London\\Experiment data\\June26\\nexus\\AAAA\\",
                obj_outpath="D:\\starlight_image_processing\\AAAA\\",
                maskx=maskx_init,
                title="AAAA")
    
    im = create_huge_image(imgs, ncols=5, nrows=5, names=momentum_list_AAAA)
    im.show()
    
    

    imgs = load_images(momentum_list=momentum_list_AAAA_ff,
                obj_path="D:\\raw_images\\AAAA\\bg",
                dark_path="D:\\raw_images\\AAAA\\46",
                nexus_path="C:\\Users\\sec90771\\OneDrive - University College London\\Experiment data\\June26\\nexus\\AAAA bg\\",
                obj_outpath="D:\\starlight_image_processing\\flatfields\\",
                maskx=maskx_init,
                title="AAAA flatfield")

    create_huge_image(imgs, ncols=5, nrows=5, names=momentum_list_AAAA)
    im.show()

    """

    imgs = load_images(momentum_list=momentum_list_9V,
                obj_path="D:\\raw_images\\9V_Toshiba",
                dark_path="D:\\raw_images\\AAAA\\46",
                nexus_path="C:\\Users\\sec90771\\OneDrive - University College London\\Experiment data\\June26\\nexus\\9V\\",
                obj_outpath="D:\\starlight_image_processing\\9V\\",
                maskx=maskx_init,
                title="9V")

    im = create_huge_image(imgs, ncols=5, nrows=5, names=momentum_list_9V)
    im.show()
    im.save("9V_raw.tif")

    imgs = load_images(momentum_list=momentum_list_9V_ff,
                obj_path="D:\\raw_images\\9V_Toshiba\\bg",
                dark_path="D:\\raw_images\\AAAA\\46",
                nexus_path="C:\\Users\\sec90771\\OneDrive - University College London\\Experiment data\\June26\\nexus\\9V bg\\",
                obj_outpath="D:\\starlight_image_processing\\flatfields\\",
                maskx=maskx_init,
                title="9V flatfield")

    im = create_huge_image(imgs, ncols=5, nrows=5, names=momentum_list_9V_ff)
    im.show()
    im.save("9V_ff_raw.tif")



