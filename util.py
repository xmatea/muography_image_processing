import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit

approximate_intensity_peak_pos, approximate_intensity_peak_sigma = 10, 100
cmap = "magma"


def gaussian(x, m, s, a):
    return a/(s * np.sqrt(2*np.pi)) * np.exp(-((x-m)**2) / (2*(s**2)))

def show_image(image, title, xlim: tuple[float, float] | None = None):
    # flatten image to get pixel intensity distribution
    flattened_img = image.flatten()
    uniq = np.unique_counts(flattened_img)

    # fit a gaussian to the intensity to get some idea of where to start
    popt, pcov = curve_fit(f=gaussian, xdata=uniq.values, ydata=uniq.counts,
                           p0=[approximate_intensity_peak_pos, approximate_intensity_peak_sigma, 100])

    # initial viewing range
    init_start = max(0, popt[0] - 5 * popt[1])
    init_stop = min(1, popt[0] + 20 * popt[1])
    init_width = init_stop - init_start

    if xlim is None:
        # set up contraints for sliders
        min_pos = max(0, init_start - 2*init_width)
        max_pos = min(1, init_stop + 2*init_width)
        max_width = min(1, init_width * 2)

    else:
        min_pos = xlim[0]
        max_pos = xlim[1]
        max_width = max_pos - min_pos

    # cut out only selected pixels
    #new_image = get_pixels_in_range(image, init_start, init_stop)
    img = np.clip(image, a_min=init_start, a_max=init_stop)

    img = img - np.min(img)

    new_image = (img / np.max(img)) * 20

    # set up plots
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle(title)

    img_plot = ax[0].imshow(new_image, cmap=cmap)

    ax[0].set_title(title)
    ax[1].set_title("Pixel brightness range")

    selected_counts = uniq.counts[(uniq.values >= min_pos) & (uniq.values <= max_pos)]
    selected_values = uniq.values[(uniq.values >= min_pos) & (uniq.values <= max_pos)]

    ax[1].plot(selected_values, selected_counts)

    ymin, ymax = ax[1].get_ylim()
    shaded_area_plot = ax[1].fill_betweenx(y=[ymin, ymax], x1=init_start, x2=init_stop, color="lightgrey", alpha=1)
    ax[1].set_xlabel("Pixel brightness (normalised)")
    ax[1].set_ylabel("Counts")

    if xlim is not None:
        ax[1].set_xlim(xlim)

    # add window position slider
    ax_window_pos = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    window_pos_slider = Slider(
        ax=ax_window_pos,
        label='Window position',
        valmin= min_pos,
        valmax=max_pos,
        valinit=init_start,
    )

    # add window size slider
    ax_window_size = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    window_size_slider = Slider(
        ax=ax_window_size,
        label='Window width',
        valmin=0.0,
        valmax=max_width,
        valinit=init_width,
    )

    def update(val):
        width = window_size_slider.val
        pos = window_pos_slider.val

        # get start and stop position from sliders
        start = pos - 0.5 * width
        stop = pos + 0.5 * width

        # update imagge
        #new_img = get_pixels_in_range(image, start, stop)

        img = np.clip(image, a_min=start, a_max=stop)
        img = img - np.min(img)

        new_img = (img / np.max(img)) * 20

        img_plot.set_data(new_img)

        # clear axes and redraw to display new range because idk how to do it properly
        ax[1].clear()
        ax[1].set_title("Pixel brightness range")
        ax[1].plot(selected_values, selected_counts)
        ymin, ymax = ax[1].get_ylim()
        shaded_area_plot = ax[1].fill_betweenx(y=[ymin, ymax], x1=start, x2=stop, color="lightgrey", alpha=1)
        ax[1].set_xlabel("Pixel brightness (normalised)")

        if xlim is not None:
            ax[1].set_xlim(xlim)
        ax[1].set_ylabel("Counts")

        fig.canvas.draw_idle()

    # register the update function with each slider
    window_size_slider.on_changed(update)
    window_pos_slider.on_changed(update)

    plt.show()