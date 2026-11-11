import re
from io import BytesIO

import numpy as np
import scipy
from PIL.ImageDraw import ImageDraw
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

from bg_corrections import correct_image
from lmfit import Parameters, minimize, report_fit
import os
import matplotlib.colors as mcolors

from timestamp_splitter_starlight import load_counts
plt.style.use('seaborn-v0_8-deep')


def flat_field_correction(img: np.ndarray, bg: np.ndarray, dark: np.ndarray | None = None):
    if dark is None:
        dark = np.zeros_like(img)

    gain = np.mean(bg) / (bg - dark)
    corr = (img - dark) * gain
    return corr

def percentile_stretching(img, cutoff_percentile: int):
    min_val = np.percentile(img, cutoff_percentile)
    mean_val = np.percentile(img, 100-cutoff_percentile)

    #return np.clip(img, a_min=0, a_max=mean_val)

    return np.clip(((img-min_val)/(mean_val-min_val)*255), a_min=0, a_max=255)
    #return np.clip(img, a_min=0, a_max=2*mean_val-min_val)

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

            text = f"{names[img_index]} MeV/c"
            draw.text((i * single_width + 10, mom * single_height + 10), text, font_size=18, fill="blue", align="left",
                      anchor="lt")

            img_index += 1

    return new_im_labelled

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).median(-1).median(1)

if __name__ == '__main__':
    data_path = "D:/starlight_image_processing/AAAA/"
    data_bg_path = "D:/starlight_image_processing/AAAA/bg/"
    dark_img_path = "D:/starlight_image_processing/dark.tif"

    nexus_path_obj = "C:\\Users\\sec90771\\OneDrive - University College London\\Experiment data\\June26\\nexus\\AAAA_\\"
    nexus_path_bright = "C:\\Users\\sec90771\\OneDrive - University College London\\Experiment data\\June26\\nexus\\AAAA_ bg\\"

    cmap = "magma"
    #cmap = "vanimo"

    colors = ["pink", "purple", "black", "blue", "green"]
    nodes = [0.0, 0.3, 0.5, 0.7, 1]
    cmap2 = mcolors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    #cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)


    img_arrays = []
    imgs = []
    filenames = []

    masky_init = [80, -150]
    maskx_init = [600, -450]


    bg_shift = [0, 0]

    shifted = [66, 64]

    delayed_counts_obj = dict(sorted(load_counts(nexus_path_obj).items()))
    delayed_counts_bright = dict(sorted(load_counts(nexus_path_bright).items()))
    #delayed_counts_dark = delayed_counts_bright[54]


    dark_img = np.array(Image.open(dark_img_path))[masky_init[0]:masky_init[1], maskx_init[0]:maskx_init[1]]

    plt.figure()
    plt.title("AAAA_")

    plt.plot(delayed_counts_bright.keys(), delayed_counts_bright.values(), "o-", ms=5, label="Bright-field")
    plt.plot(delayed_counts_obj.keys(), delayed_counts_obj.values(), "o-", ms=5, label="Object")

    plt.xlabel("Momentum")
    plt.ylabel("Delayed counts")
    plt.legend()
    plt.show()

    try:
        os.mkdir(data_path + "outputs")
    except FileExistsError:
        pass

    for filename in os.listdir(data_path):
        if not filename.endswith(".tif"):
            continue

        match = re.search(r"(\d+.?\d+)", filename)
        if match is None:
            continue

        momentum = float(match.group(1))

        img_path = data_path + filename
        bg_path = data_bg_path + filename

        img_counts = delayed_counts_obj[momentum]
        bg_counts = delayed_counts_bright[momentum]

        masky = masky_init
        maskx = maskx_init

        if momentum in shifted:
            masky[0] += bg_shift[0]
            masky[1] += bg_shift[0]

            maskx[0] += bg_shift[1]
            maskx[1] += bg_shift[1]

        raw_img = (np.array(Image.open(img_path))[masky[0]:masky[1], maskx[0]:maskx[1]]) / img_counts *1e5
        bg_img = (np.array(Image.open(bg_path))[masky[0]:masky[1], maskx[0]:maskx[1]]) / bg_counts *1e5

        #raw_img = rebin(raw_img, shape=(raw_img.shape[0]//2, raw_img.shape[1]//2))
        #bg_img = rebin(bg_img, shape=(bg_img.shape[0]//2, bg_img.shape[1]//2))

        #dark_img = rebin(dark_img, shape=(dark_img.shape[0]//2, dark_img.shape[1]//2))

        # https://en.wikipedia.org/wiki/Flat-field_correction
        gain = np.nan_to_num(np.mean(bg_img) / (bg_img), nan=0)
        #corr_img = (raw_img - dark_img) * gain

        #corr_img = np.nan_to_num(raw_img / bg_img) - 1
        img = np.nan_to_num((raw_img / bg_img), nan=0)
        img = percentile_stretching(img, 1)

        img = scipy.ndimage.median_filter(img, size=3)

        #norm = mcolors.TwoSlopeNorm(vmin=np.min(img), vmax=np.max(img), vcenter=1)

        #fig, ax = plt.subplots(ncols=1)
        #ax[0].imshow(img, cmap=cmap)
        #plt.show()

        """
        stretched_img = scipy.ndimage.median_filter(stretched_img, size=3)

        np.nan_to_num(stretched_img, copy=False, nan=0)

        # correct background of image
        corr = correct_image(stretched_img, offs=10, intensity=3000, maskx=(mask_aaaa[0][0]-200, mask_aaaa[1][0]-200),
                      masky=(mask_aaaa[0][1]-50, mask_aaaa[1][1]-50), plot=True)

        corr_img = np.nan_to_num(stretched_img / corr, nan=0)
        corr_img = corr_img / np.max(corr_img) * 255
        
        """

        img_arrays.append(img)
        filenames.append(filename)

        print("processed", filename)

        plt.figure()
        plt.imshow(img, cmap=cmap)
        plt.axis('off')

        bytes = BytesIO()
        plt.savefig(bytes, bbox_inches='tight', pad_inches=0)
        plot_image = Image.open(bytes)


        # SAVE INDIVIDUAL IMAGES
        #plot_image.save(data_path + f"outputs/{filename[:-4]}.png")

        plt.close()
        mom = f"{filename[4:6]}MeV/c"

        draw = ImageDraw(plot_image)
        draw.text((10, 10), mom, font_size=28, fill="white", align="left", anchor="lt")

        imgs.append(plot_image)

    # sort filenames in case files are not loaded alphabetically!
    a = np.array(filenames)
    p = a.argsort()

    filenames_sorted = a[p]
    imgs_sorted = [imgs[index] for index in p]

    # SAVE COMPARISON IMAGE
    final_img = create_huge_image(imgs_sorted, nrows=2, ncols=8, names=filenames_sorted)
    final_img.show()
    #final_img.save(data_bg_path + "outputs/summary_raw.png")

    # SAVE GIF
    """
    imgs_sorted[0].save(
        f"{data_path}outputs/summary.gif",
        save_all=True,
        append_images=imgs_sorted[1:],
        duration=300,
        loop=0,
        optimize=False
    )
    """







