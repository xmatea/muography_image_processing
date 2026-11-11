from io import BytesIO

import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
from lmfit import Parameters, minimize, report_fit
from matplotlib import pyplot as plt

from process import create_huge_image, percentile_stretching, open_beam_normalisation, fit_2d_gaussian, \
    percentile_stretching_2, fit_2d_gaussian_masked

if __name__ == "__main__":
    folder = "D:\\starlight_image_processing\\outputs\\AAAA_2\\"

    #momentum_list = [64, 66]
    momentum_list = [44, 48, 50, 52, 54, 56, 58, 60, 62, 68, 70, 72, 74]

    bg_momentum_list = np.arange(76, 92, 2)

    imgstack = []
    names = []
    cmap = "berlin"

    plot = True
    save = False

    gifsave = True
    output_path = "D:\starlight_image_processing\outputs\AAAA_2\\"

    # open beam normalisation on side edge
    p1 = (116, 6)
    p2 = (467, 1000)

    for i, momentum in enumerate(momentum_list):
        path = folder + f"{momentum}_MED.tif"
        bg_path = folder + f"bg//{momentum}_MED.tif"

        try:
            img = np.array(Image.open(path))
            bg = np.array(Image.open(bg_path))

            img, bg = open_beam_normalisation(img, bg, 1e6, p1, p2)
            #fitted_bg = fit_2d_gaussian(bg, intensity=10, offset=0, plot=True)

            maskx = (150, -120)
            masky = (10, -10)


            if momentum in [64, 66]:
                fitted_bg = fit_2d_gaussian_masked(img, intensity=10, offset=0, xpos=273, ypos=534, plot=True, maskx=maskx, masky=masky)
            else:
                fitted_bg = fit_2d_gaussian_masked(img, intensity=10, offset=0, plot=True, maskx=maskx, masky=masky)
                #fitted_bg = fit_2d_gaussian(img, intensity=10, offset=0, plot=True)


            corr = np.nan_to_num(img / fitted_bg, nan=1)
            corr = percentile_stretching_2(corr, perc=3, overflow=1)

            final_mask_x = [100, -100]
            final_mask_y = [10, -10]

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

            plot_image = Image.open(bytes)

            if save:

                plot_image.save("C:\\Users\\sec90771\\OneDrive - University College London\\Pictures"
                                + f"\\AAAA images\\gaussian_fitting\\processed_magma\\{momentum}.png")

            draw = ImageDraw(plot_image)

            label = f"{momentum} MeV/c"

            draw.text((10, 10), label, font_size=18, fill="white", align="left", anchor="lt")

            if save:
                plot_image.save("C:\\Users\\sec90771\\OneDrive - University College London\\Pictures"
                +f"\\AAAA images\\gaussian_fitting\\processed_magma\\{momentum}_labelled.png")

            imgstack.append(plot_image)
            names.append(momentum)

        except FileNotFoundError:
            print(f"{path} not found")


    final_img = create_huge_image(imgstack, nrows=2, ncols=8, names=names)
    final_img.show()
    if save:
        final_img.save("C:\\Users\\sec90771\\OneDrive - University College London"
        + "\\Pictures\\AAAA images\\gaussian_fitting\\AAAA_magma.png")

    if gifsave:
        imgstack[0].save(
            f"C:\\Users\\sec90771\\OneDrive - University College London"
        + "\\Pictures\\AAAA images\\gaussian_fitting\\summary_berlin.gif",
            save_all=True,
            append_images=imgstack[1:],
            duration=300,
            loop=0,
            optimize=False
        )
