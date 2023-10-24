from numpy import np

def calculate_resolution_array(max_res_area=512 ** 2, bucket_lower_bound_res=256, rounding=64):
    """
    helper function to calculate image bucket 

    Parameters:
    - max_res_area (int): The maximum target resolution area of the image.
    - bucket_lower_bound_res (int): minimum minor axis (smaller axis).
    - rounding (int): rounding steps / rounding increment.

    Returns:
    - resolution (numpy.ndarray): A 2D NumPy array representing the resolution pairs (width, height).
    """
    root_max_res = max_res_area ** (1 / 2)
    centroid = int(root_max_res)

    # a sequence of number that divisible by 64 with constraint
    w = np.arange(bucket_lower_bound_res // rounding * rounding, centroid // rounding * rounding + rounding, rounding)
    # y=1/x formula with rounding down to the nearest multiple of 64
    # will maximize the clamped resolution to maximum res area
    h = ((max_res_area / w) // rounding * rounding).astype(int)

    # is square array possible? if so chop the last bit before combining
    if w[-1] - h[-1] == 0:
        w_delta = np.flip(w[:-1])
        h_delta = np.flip(h[:-1])

    w = np.concatenate([w,w_delta])
    h = np.concatenate([h,h_delta])

    resolution = np.stack([w,h]).T

    return resolution