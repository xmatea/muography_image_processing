from io import BytesIO

import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
from lmfit import Parameters, minimize, report_fit
from matplotlib import pyplot as plt

from process import create_huge_image, percentile_stretching, open_beam_normalisation, fit_2d_gaussian, \
    percentile_stretching_2, fit_2d_gaussian_masked

if __name__ == "__main__":
    folder = "D:\\starlight_image_processing\\outputs\\9V\\"
    bg_folder = "D:\\starlight_image_processing\\bgs\\9V\\"

    momentum_list = np.arange(56, 92, 2)
    bg_momentum_list = momentum_list

    rawstack = []
    imgstack = []
    bgstack = []

    names = []
    cmap = "magma"

    plot = True
    save = True

    gifsave = True
    output_path = "D:\\starlight_image_processing\\outputs\\9V\\"
    save_dest = "C:\\Users\\sec90771\\OneDrive - University College London\\Pictures\\9V images\\gaussian fitted\\"
    use_saved_bg = False

    # open beam normalisation on side edge
    p1 = (220, 30)
    p2 = (670, 932)

    n_muons_arr = []
    n_muons_arr_post = []
    n_bg_arr = []
    n_bg_arr_post = []
    momenta = []

    """
    for i, momentum in enumerate(bg_momentum_list):
        maskx = (500, 501)
        masky = (500, 501)

        bg_path = folder + f"bg//{momentum}_MED.tif"
        img =  np.array(Image.open(bg_path))
        img = img[:-100, :]

        fitted_bg = fit_2d_gaussian_masked(img, intensity=10, offset=0, plot=plot, maskx=maskx, masky=masky, ypos=500)
        np.save(f"{output_path}flatfields//{momentum}.npy", fitted_bg)
    """

    for i, momentum in enumerate(momentum_list):
        path = folder + f"{momentum}_MED.tif"
        bg_path = f"{bg_folder}//{momentum}_MED.tif"

        try:
            img_raw = np.array(Image.open(path))
            img = img_raw[:-130, :]

        except FileNotFoundError:
            print(f"{path} not found")
            continue

        print(f"processing {path}")
        #fitted_bg = fit_2d_gaussian(bg, intensity=10, offset=0, plot=True)

        maskx = (230, 670)
        masky = (30, 932)


        #fitted_bg = np.array(Image.open(f"{bg_folder}{momentum}_MED.tif"))
        fitted_bg = np.load(f"D:\\starlight_image_processing\\bgs\\fitted\\{momentum}.npy")
        #fitted_bg = fitted_bg[:-130, :]

        """
        elif use_saved_bg:
            bg_path = f"{bg_folder}//{momentum}.npy"
            fitted_bg = np.load(bg_path)
        else:
            fitted_bg = fit_2d_gaussian_masked(img, intensity=10, offset=0, plot=plot, maskx=maskx, masky=masky, ypos=500)
            np.save(f"{output_path}bg//{momentum}.npy", fitted_bg)
        """
        """
        _, _, n_muons, n_bg = open_beam_normalisation(img, fitted_bg, 1e3, p1, p2, plot=True)
        _, _ = open_beam_normalisation(img, bg, 1e3, p1, p2, plot=True)
        

        n_muons_arr.append(n_muons)
        n_muons_arr_post.append(n_muons_post)
        n_bg_arr.append(n_bg)
        n_bg_arr_post.append(n_bg_post)

        momenta.append(momentum)
        """

        bg = fitted_bg

        corr = np.nan_to_num(img / bg, nan=0)
        corr = percentile_stretching(corr, cutoff_percentile=2)
        #corr = percentile_stretching_2(corr, perc=2, overflow=1)

        final_mask_x = [150, -150]
        final_mask_y = [1, -1]

        # apply final mask
        corr = corr[final_mask_y[0]:final_mask_y[1], final_mask_x[0]:final_mask_x[1]]

        if plot:
            fig, ax = plt.subplots(ncols=3)
            fig.suptitle(f"momentum={momentum}")
            ax[0].imshow(img)
            ax[0].set_title("object image")
            ax[1].imshow(bg)
            ax[1].set_title("bright field")
            ax[2].imshow(corr)
            ax[2].set_title("correction")

            plt.show()

        plt.imshow(corr, cmap=cmap)
        plt.axis('off')

        bytes = BytesIO()
        plt.savefig(bytes, bbox_inches='tight', pad_inches=0)
        plt.close()

        corr_image = Image.open(bytes)

        plt.imshow(img_raw, cmap=cmap)
        plt.axis('off')

        bytes = BytesIO()
        plt.savefig(bytes, bbox_inches='tight', pad_inches=0)
        plt.close()

        raw_image = Image.open(bytes)

        plt.imshow(bg, cmap=cmap)
        plt.axis('off')

        bytes = BytesIO()
        plt.savefig(bytes, bbox_inches='tight', pad_inches=0)
        plt.close()

        bg_image = Image.open(bytes)

        if save:
            #corr_image.save("C:\\Users\\sec90771\\OneDrive - University College London\\Pictures"
                            #+ f"\\AAAA images\\gaussian_fitting\\processed_magma\\{momentum}.png")
            pass

        draw = ImageDraw(corr_image)
        draw_raw = ImageDraw(raw_image)
        draw_bg = ImageDraw(bg_image)

        label = f"{momentum} MeV/c"

        draw.text((10, 10), label, font_size=18, fill="white", align="left", anchor="lt")
        draw_raw.text((10, 10), label, font_size=18, fill="white", align="left", anchor="lt")
        draw_bg.text((10, 10), label, font_size=18, fill="white", align="left", anchor="lt")

        if save:
            #corr_image.save("C:\\Users\\sec90771\\OneDrive - University College London\\Pictures"
            #+f"\\AAAA images\\gaussian_fitting\\processed_magma\\{momentum}_labelled.png")
            pass

        imgstack.append(corr_image)
        rawstack.append(raw_image)
        bgstack.append(bg_image)
        names.append(momentum)


    final_img = create_huge_image(imgstack, nrows=3, ncols=6, names=names)
    final_img.show()

    final_img_raw = create_huge_image(rawstack, nrows=3, ncols=6, names=names)
    final_img_bg = create_huge_image(bgstack, nrows=3, ncols=6, names=names)

    """
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(momenta, n_muons_arr, label="muons pre norm")
    ax[1].plot(momenta, n_muons_arr_post, "--", label="muons post norm")
    ax[0].plot(momenta, n_bg_arr, label="bg pre norm")
    ax[1].plot(momenta, n_bg_arr_post, "--", label="bg post norm")
    ax[0].set_xlabel("Momentum (MeV/c)")
    ax[1].set_xlabel("Momentum (MeV/c)")
    ax[0].set_ylabel("Counts (arb)")
    ax[1].set_ylabel("Counts (arb)")

    ax[0].vlines(76, ymin=0, ymax=np.max(n_muons_arr)*1.2)
    ax[1].vlines(76, ymin=0, ymax=np.max(n_muons_arr_post)*1.2)
    plt.legend()
    plt.show()
    """

    if save:
        final_img.save(f"{save_dest}9V_{cmap}_scaled_darkcorr_skewgaussian.png")
        final_img_raw.save(f"{save_dest}9V_{cmap}_scaled_darkcorr_raw.png")
        final_img_bg.save(f"{save_dest}9V_{cmap}_skewgaussian_bg.png")

    if gifsave:
        gif_ims = []
        for im in imgstack:
            gif_ims.append(im.quantize(colors=255, method=Image.Quantize.FASTOCTREE, dither=0))

        imgstack[0].save(f"{save_dest}9V_{cmap}_skewgaussian.gif",
            save_all=True,
            append_images=imgstack[1:],
            duration=300,
            loop=0,
            optimize=False
        )
