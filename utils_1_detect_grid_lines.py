import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks

def detect_grid_lines(binary_roi, axis, expected_lines, offset, debug_path=None):
    """
    Detect grid lines using projection profiles on ROI.

    Parameters
    ----------
    binary_roi : ndarray
        Cropped binary image
    axis : str
        "vertical" or "horizontal"
    expected_lines : int
        Number of grid lines expected (4)
    offset : int
        Offset to convert ROI coordinates back to original image
    debug_path : str
        Where to save projection debug plot

    Returns
    -------
    np.array
        Detected line positions in ORIGINAL image coordinates
    """

    if axis == "vertical":
        projection = np.sum(binary_roi, axis=0)
    else:
        projection = np.sum(binary_roi, axis=1)

    projection = projection.astype(float)

    if projection.max() > 0:
        projection /= projection.max()

    peaks, _ = find_peaks(
        projection,
        distance=len(projection)//10,
        prominence=0.2
    )

    if len(peaks) > expected_lines:
        peaks = peaks[np.argsort(projection[peaks])[-expected_lines:]]

    peaks = np.sort(peaks)

    # convert back to original coordinates
    peaks = peaks + offset

    if debug_path is not None:

        plt.figure(figsize=(10,4))
        plt.plot(projection)

        plt.scatter(
            peaks - offset,
            projection[peaks - offset],
            color="red",
        )

        plt.title(f"{axis} projection (ROI)")
        plt.savefig(debug_path)
        plt.close()

    return peaks
