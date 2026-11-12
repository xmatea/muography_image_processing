from io import BytesIO

import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
from matplotlib import pyplot as plt

from process import create_huge_image, percentile_stretching, fit_2d_gaussian, fit_2d_gaussian_skewed
from util import show_image

if __name__ == "__main__":
    toshiba_path = "flatfields/fitted_standard_gauss/9V\\"
    aaaa_path = "flatfields/fitted_standard_gauss/AAAA\\"
    ff_path = "flatfields\\"
    save_dest = "C:\\Users\\sec90771\\OneDrive - University College London\\Pictures\\9V and AAAA flatfields"
    cmap = "magma"

    toshiba_momenta = np.arange(76, 92, 2)
    aaaa_momenta = [44, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74]

    full_range = np.arange(44, 92, 2)

    imgstack = []

    momenta = []
    xpos = []
    ypos = []
    skew = []
    sigmax = []
    sigmay = []

    shifts = []

    AAAA_images = []
    T9V_images = []
    shift_imgs = []
    fitted_bgs = []
    params = []


    imgstack = []

    upper = 620
    lower = 493

    dx = upper - lower

    momenta = []
    n_muons = []
    n_bg = []

    for mom in full_range:
        filepath = f"{ff_path}{mom}_MED.tif"

        try:
            ff = np.array(Image.open(filepath))
        except FileNotFoundError:
            print(f"data not found at {filepath}")
            continue

        if mom == 54:
            continue

        #ff = percentile_stretching(ff, cutoff_percentile=1)

        print("starting fit")
        curve, params = fit_2d_gaussian(ff, 10, 1, ypos=700, plot=True, vary_centroid=True)
        np.save(f"flatfields\\fitted_standard_gauss\\{mom}.npy", curve)

        momenta.append(mom)
        ypos.append(params["centroid_y"])
        sigmax.append(params["sigma_x"])
        sigmay.append(params["sigma_y"])
        fitted_bgs.append(curve)
        shifts.append(params["centroid_y"] - lower)

        if mom <= 74:
            diff = int(params["centroid_y"] - lower)
            img = ff[diff:, :]

            plt.imshow(img)
            plt.show()
            print(img.shape)

        else:
            img = ff

        shift_imgs.append(img)

    fig, ax = plt.subplots(ncols=4)
    ax[0].plot(momenta, ypos, label="y pos")
    ax[0].set_title("y pos")
    #ax[1].plot(momenta, skew, label="skew")
    #ax[1].set_title("skew")
    ax[2].plot(momenta, sigmax, label="sigmax")
    ax[2].set_title("sigmax")
    ax[3].plot(momenta, sigmay, label="sigmay")
    ax[3].set_title("sigmay")
    plt.show()

    y_max = np.argmax(shifts)
    min_height = shift_imgs[y_max].shape[0]
    target_dims = (min_height, shift_imgs[y_max].shape[1])

    print()
    for i, img in enumerate(shift_imgs):
        height, width = img.shape

        dy = height - min_height

        print("before corr:", img.shape)
        # make sure all images have the same dimensions
        if dy:
            img = img[:-dy, :]

        print("after corr:", img.shape)

        Image.fromarray(img).save(f"{toshiba_path}{momenta[i]}_MED.tif")

        plt.imshow(img, cmap=cmap)
        plt.axis('off')

        bytes = BytesIO()
        plt.savefig(bytes, bbox_inches='tight', pad_inches=0)
        plt.close()

        img = Image.open(bytes)
        draw = ImageDraw(img)

        label = f"{momenta[i]} MeV/c"
        draw.text((10, 10), label, font_size=18, fill="white", align="left", anchor="lt")

        imgstack.append(img)

    final_img = create_huge_image(imgstack, nrows=5, ncols=5, names=np.zeros_like(imgstack))
    final_img.save(f"{save_dest}\\bg_{cmap}.png")
    final_img.show()

    """
    for i, mom in enumerate(full_range):

        img_arr = imgarr[:-130, :]
        Image.fromarray(imgarr).save(f"{toshiba_path}{mom}_MED.tif")

        plt.imshow(imgarr, cmap=cmap)
        plt.axis('off')

        bytes = BytesIO()
        plt.savefig(bytes, bbox_inches='tight', pad_inches=0)
        plt.close()

        img = Image.open(bytes)
        draw = ImageDraw(img)

        label = f"{mom} MeV/c"
        draw.text((10, 10), label, font_size=18, fill="white", align="left", anchor="lt")

        shift_imgs.append(img)
        print(img.size)
        
    """

    """
    for mom in full_range:
        filepath = f"{ff_path}{mom}_MED.tif"

        try:
            ff = np.array(Image.open(filepath))
            ff = percentile_stretching(ff, cutoff_percentile=1)
        except FileNotFoundError:
            print(f"data not found at {filepath}")
            continue

        plt.imshow(ff, cmap=cmap)
        plt.axis('off')

        bytes = BytesIO()
        plt.savefig(bytes, bbox_inches='tight', pad_inches=0)
        plt.close()

        img = Image.open(bytes)
        draw = ImageDraw(img)

        label = f"{mom} MeV/c"
        draw.text((10, 10), label, font_size=18, fill="white", align="left", anchor="lt")
        imgstack.append(img)

    final_img = create_huge_image(imgstack, nrows=5, ncols=5, names=np.zeros_like(imgstack))
    final_img.save(f"{save_dest}\\bg_{cmap}.png")
    final_img.show()
    """



