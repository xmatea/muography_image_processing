import numpy as np
from matplotlib import pyplot as plt
from lmfit import Parameters, minimize, report_fit

def correct_image(img, offs, intensity, maskx, masky, plot=False):
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

    initial = Parameters()
    initial.add("height", value=intensity, min=0)
    initial.add("background", value=offs, vary=True)
    initial.add("centroid_x", value=masked_corr.shape[1] / 2, vary=True)
    initial.add("centroid_y", value=masked_corr.shape[0] / 2, vary=True)
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

        plt.figure()
        plt.plot(masked_corr[300, :], label="Data")
        plt.plot(fit_curve_raw[300, :], label="Fit")
        plt.show()

    return fit_curve