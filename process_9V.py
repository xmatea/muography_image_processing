import numpy as np
import scipy

from matplotlib import pyplot as plt
from PIL import Image

import matplotlib.colors as mcolors

from process import load_folder, percentile_stretching
from timestamp_splitter_starlight import load_counts


if __name__ == '__main__':
    obj_path = "D:\\raw_images\\9V_Toshiba"
    ff_path = "D:\\raw_images\\9V_Toshiba\\bg"
    dark_img_path = "D:\\raw_images\\AAAA\\46"
    output_path = "D:\\starlight_image_processing\\outputs\\9V\\"
    ff_output_path = "D:\\starlight_image_processing\\bgs\\"

    nexus_path_obj = "C:\\Users\\sec90771\\OneDrive - University College London\\Experiment data\\June26\\nexus\\9V\\"
    nexus_path_bright = "C:\\Users\\sec90771\\OneDrive - University College London\\Experiment data\\June26\\nexus\\9V bg\\"

    cmap = "magma"

    colors = ["pink", "purple", "black", "blue", "green"]
    nodes = [0.0, 0.3, 0.5, 0.7, 1]
    cmap2 = mcolors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    #cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)

    img_arrays = []
    imgs = []
    filenames = []

    masky_init = [0, -0]
    maskx_init = [300, -200]

    delayed_counts_obj = dict(sorted(load_counts(nexus_path_obj).items()))
    delayed_counts_bright = dict(sorted(load_counts(nexus_path_bright).items()))
    #delayed_counts_dark = delayed_counts_bright[54]

    #dark_img = load_folder(dark_img_path, show_image=True, maskx=maskx_init)

    plt.figure()
    plt.title("9V")

    plt.plot(delayed_counts_bright.keys(), delayed_counts_bright.values(), "o-", ms=5, label="Bright-field")
    plt.plot(delayed_counts_obj.keys(), delayed_counts_obj.values(), "o-", ms=5, label="Object")

    plt.xlabel("Momentum")
    plt.ylabel("Delayed counts")
    plt.legend()
    plt.show()

    momentum_list = np.arange(56, 92, 2)
    bg_momentum_list = np.arange(76, 92, 2)

    dark_img = load_folder(dark_img_path, show_image=False, maskx=maskx_init)

    for momentum in momentum_list:
        #show_image(dark_img, "dark", (0, 255))

        try:
            obj_folder = obj_path + f"\\{momentum}"
            ff_folder = ff_path + f"\\{momentum}"

            obj_img = (load_folder(obj_folder, show_image=False, maskx=maskx_init) / delayed_counts_obj[momentum]) * 1e6
            #show_image(obj_img, "obj", (0, 255))

            n_target = np.sum(obj_img[:, 73]) + np.sum(obj_img[:, 831])
            n_dark = np.sum(dark_img[:, 73]) + np.sum(dark_img[:, 831])

            obj_img = (obj_img / n_target) * n_dark
            obj_img = obj_img #- dark_img

            obj_img = percentile_stretching(obj_img, cutoff_percentile=1)
            #obj_img = scipy.signal.medfilt(obj_img, 3)

            Image.fromarray(obj_img).save(f"{output_path}{momentum}_MED.tif")

            if momentum in bg_momentum_list:
                ff_img = (load_folder(ff_folder, show_image=False, maskx=maskx_init) / delayed_counts_bright[momentum]) * 1e6
                # show_image(obj_img, "obj", (0, 255))

                n_target = np.sum(ff_img[:, 73]) + np.sum(ff_img[:, 831])
                n_dark = np.sum(dark_img[:, 73]) + np.sum(dark_img[:, 831])

                ff_img = (ff_img / n_target) * n_dark
                ff_img = ff_img #- dark_img

                ff_img = percentile_stretching(ff_img, cutoff_percentile=1)
                #ff_img = scipy.signal.medfilt(ff_img, 3)
                Image.fromarray(ff_img).save(f"{ff_output_path}{momentum}_MED.tif")

            """
            obj_img, L, S = run_pca(obj_img-dark_img, show_image=True)
            Image.fromarray(L).save(f"{output_path}{momentum}_low_rank.tif")
            Image.fromarray(S).save(f"{output_path}{momentum}_sparse.tif")
            """

            """
            ff_img, L, S = run_pca(ff_img-dark_img, show_image=True)
            Image.fromarray(L).save(f"{output_path}bg\\{momentum}_low_rank.tif")
            Image.fromarray(S).save(f"{output_path}bg\\{momentum}_sparse.tif")
            """

        except FileNotFoundError as e:
            print(str(momentum) + f" data not found at {obj_folder}")
            continue