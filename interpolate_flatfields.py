from io import BytesIO

import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
from matplotlib import pyplot as plt

from process import create_huge_image, percentile_stretching, fit_2d_gaussian, fit_2d_gaussian_skewed
from util import show_image

if __name__ == "__main__":
    toshiba_path = "D:\\starlight_image_processing\\bgs\\9V\\"
    aaaa_path = "D:\\starlight_image_processing\\bgs\\AAAA\\"
    ff_path = "D:\\starlight_image_processing\\bgs\\"
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


    datas = []

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

        #ff = percentile_stretching(ff, cutoff_percentile=1)

        print("starting fit")
        curve, params = fit_2d_gaussian_skewed(ff, 10, 1, 0, -2.5, ypos=600, plot=True)
        np.save(f"D:\\starlight_image_processing\\bgs\\fitted_fixed_centroid\\{mom}.npy", curve)

        momenta.append(mom)
        ypos.append(params["centroid_y"])
        skew.append(params["skew_y"])
        sigmax.append(params["sigma_x"])
        sigmay.append(params["sigma_y"])
        fitted_bgs.append(curve)

        shifts.append(params["centroid_y"] - lower)

        if mom == 54:
            continue

        if mom <= 74:
            dx = int(params["centroid_y"] - lower)
            imgarr = np.zeros_like(ff)
            imgarr[:-dx, :] = ff[dx:, :]

        else:
            plt.imshow(ff, cmap=cmap)
            plt.axis('off')

            img_arr = ff[:-130, :]

            bytes = BytesIO()
            plt.savefig(bytes, bbox_inches='tight', pad_inches=0)
            plt.close()

            img = Image.open(bytes)
            draw = ImageDraw(img)

            label = f"{mom} MeV/c"
            draw.text((10, 10), label, font_size=18, fill="white", align="left", anchor="lt")

            Image.fromarray(ff).save(f"{toshiba_path}{mom}_MED.tif")
            shift_imgs.append(img)

    final_img = create_huge_image(shift_imgs, nrows=5, ncols=5, names=np.zeros_like(shift_imgs))
    #final_img = create_huge_image(fitted_bgs, nrows=5, ncols=5, names=np.zeros_like(shift_imgs))

    final_img.save(f"{save_dest}\\bg_{cmap}.png")
    final_img.show()

    """
        fit, res = fit_2d_gaussian(newim, 10, 0, plot=True)

        x = res["centroid_x"]
        y = res["centroid_y"]
        xpos.append(x)
        ypos.append(y)
        momenta.append(mom)
        """

    fig, ax = plt.subplots(ncols=4)
    ax[0].plot(momenta, ypos, label="y pos")
    ax[0].set_title("y pos")
    ax[1].plot(momenta, skew, label="skew")
    ax[1].set_title("skew")
    ax[2].plot(momenta, sigmax, label="sigmax")
    ax[2].set_title("sigmax")
    ax[3].plot(momenta, sigmay, label="sigmay")
    ax[3].set_title("sigmay")
    plt.show()


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



