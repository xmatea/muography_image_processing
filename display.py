from io import BytesIO

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from process import create_huge_image, percentile_stretching
from process import load_folder


def percentile_stretching_2(img, perc):
    mean_val = np.percentile(img, 100 - perc)

    img_copy = (img)
    img_copy[img_copy > mean_val] = 1

    return img_copy


if __name__ == "__main__":
    folder = "D:\\starlight_image_processing\\outputs\\AAAA_2\\"

    momentum_list = [44, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74]

    imgstack = []
    names = []

    plot = True
    output_path = "D:\starlight_image_processing\outputs\AAAA_2\\"

    """
    dark_img, L, S = load_folder(dark_img_path, show_image=False, maskx=maskx, dark=None)

    Image.fromarray(dark_img).save(f"{output_path}dark\\dark_MED.tif")
    Image.fromarray(L).save(f"{output_path}dark\\dark_low_rank.tif")
    Image.fromarray(S).save(f"{output_path}dark\\dark_sparse.tif")
    """

    for i, momentum in enumerate(momentum_list):
        path = folder + f"{momentum}_MED.tif"
        bg_path = folder + f"bg//{momentum}_MED.tif"

        try:
            img = np.array(Image.open(path))
            img = img - np.min(img)

            #open beam normalisation on side edge
            x_min, y_min = (116, 6)
            x_max, y_max = (467, 1000)

            mask_copy = np.copy(img)
            mask_copy[y_min:y_max, x_min:x_max] = 0

            n_muons = np.sum(mask_copy)

            if i == 0:
                norm = n_muons
                #plt.imshow(mask_copy, cmap="magma")
                #plt.show()

            else:
                img = (img / n_muons) * norm

            bg = np.array(Image.open(bg_path))
            bg = bg - np.min(bg)

            mask_copy_bg = np.copy(img)
            mask_copy_bg[y_min:y_max, x_min:x_max] = 0

            n_bg = np.sum(mask_copy_bg)
            bg = (bg / n_bg) * norm

            print(bg.shape)

            #corr = percentile_stretching(np.nan_to_num(img / bg, nan=0), cutoff_percentile=3)
            corr = percentile_stretching_2(np.nan_to_num(img / bg, nan=0), perc=3)

            mask_copy_corr = np.copy(corr)
            mask_copy_corr[y_min:y_max, x_min:x_max] = 0

            n_corr = np.sum(mask_copy_corr)
            corr = (corr / n_corr) * norm

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

            plt.figure()
            plt.imshow(corr, cmap="magma")
            plt.axis('off')

            bytes = BytesIO()
            plt.savefig(bytes, bbox_inches='tight', pad_inches=0)

            plot_image = Image.open(bytes)

            imgstack.append(plot_image)
            names.append(str(momentum))

            plt.close()

        except FileNotFoundError:
            print(f"{path} not found")


    final_img = create_huge_image(imgstack, nrows=2, ncols=8, names)
    final_img.show()
    final_img.save("C:\\Users\\sec90771\\OneDrive - University College London"
    + "\\Pictures\\AAAA images\\median then stretching\\1st_percentile_twice.png")