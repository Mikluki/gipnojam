import random

import numpy as np
from perlin_numpy import generate_fractal_noise_2d, generate_perlin_noise_2d


def meters_to_idx(meters, length_per_pixel=3e-5):
    return int(np.round(meters / length_per_pixel))


def fr2lambda(freq_hz, print_=False):
    c = 3e8
    lambda_ = c / freq_hz
    if print_ is True:
        print(lambda_)
        # print(f"{freq_hz*1e-9:3.1f} (Ghz) to {lambda_:2.2e} (m)")
    return lambda_


def zero_cross(a):
    return np.where(np.diff(np.sign(a)))[0]


def perlin_noise(shape, period, threshold=0.6, **kwargs):  # pyright: ignore
    """
    The function generate_perlin_noise_2d generates a 2D texture of perlin noise. Its parameters are:

    - shape: shape of the generated array (tuple of 2 ints)
    - period: number of periods of noise to generate along each axis (tuple of 2 ints)
    - tileable: if the noise should be tileable along each axis (tuple of 2 bools)
    Note: shape must be a multiple of period
    """
    arr = generate_perlin_noise_2d(shape=shape, res=period)
    # Map values to the range 0 to 1
    arr = (arr + 1) / 2
    # Round values based on the threshold
    arr = np.where(arr >= threshold, 1, 0)
    return arr


def fractal_noise(shape, period, octaves, lacunarity, threshold=0.6):
    """
    The function generate_fractal_noise_2d combines several octaves of 2D perlin noise to make 2D fractal noise. Its parameters are:

    - shape: shape of the generated array (tuple of 2 ints)
    - period: number of periods of noise to generate along each axis (tuple of 2 ints)
    - octaves: number of octaves in the noise (int)
    - persistence: scaling factor between two octaves (float)
    - lacunarity: frequency factor between two octaves (float)
    - tileable: if the noise should be tileable along each axis (tuple of 2 bools)
    Note: shape must be a multiple of lacunarity^(octaves-1)*period
    """
    arr = generate_fractal_noise_2d(
        shape=shape, res=period, octaves=octaves, lacunarity=lacunarity
    )
    # Map values to the range 0 to 1
    arr = (arr + 1) / 2
    # Round values based on the threshold
    arr = np.where(arr >= threshold, 1, 0)
    return arr


# def bin_matrix_fold4symmetry(x_ini):
#     x1 = np.flip(x_ini, axis=0)
#     x2 = np.flip(x_ini, axis=1)
#     x3 = np.flip(x2, axis=0)
#     return np.hstack([np.vstack([x2, x3]), np.vstack([x_ini, x1])])


def expand_right(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
) -> np.ndarray:
    dim = (x.shape[0], 1)
    noise = noise_obj.get_noise(x.shape).astype(int)
    delta = np.hstack([np.zeros(dim, dtype=int), x[:, :-1]]).astype(int)
    x = (noise & delta) | x.astype(int)
    if mask_cut is not None:
        x = apply_mask_cut(x, mask_cut=mask_cut)
    if mask_metal is not None:
        x = x.astype(int) | mask_metal.astype(int)  # pyright: ignore
    return x


def expand_left(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
) -> np.ndarray:
    dim = (x.shape[0], 1)
    noise = noise_obj.get_noise(x.shape).astype(int)
    delta = np.hstack([x[:, 1:], np.zeros(dim, dtype=int)]).astype(int)
    x = (noise & delta) | x.astype(int)
    if mask_cut is not None:
        x = apply_mask_cut(x, mask_cut=mask_cut)
    if mask_metal is not None:
        x = x.astype(int) | mask_metal.astype(int)  # pyright: ignore
    return x


def expand_up(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
) -> np.ndarray:
    dim = (1, x.shape[1])
    noise = noise_obj.get_noise(x.shape).astype(int)
    delta = np.vstack([x[1:, :], np.zeros(dim, dtype=int)]).astype(int)
    x = (noise & delta) | x.astype(int)
    if mask_cut is not None:
        x = apply_mask_cut(x, mask_cut=mask_cut)
    if mask_metal is not None:
        x = x.astype(int) | mask_metal.astype(int)  # pyright: ignore
    return x


def expand_down(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
) -> np.ndarray:
    dim = (1, x.shape[1])
    noise = noise_obj.get_noise(x.shape).astype(int)
    delta = np.vstack([np.zeros(dim, dtype=int), x[:-1, :]]).astype(int)
    x = (noise & delta) | x.astype(int)
    if mask_cut is not None:
        x = apply_mask_cut(x, mask_cut=mask_cut)
    if mask_metal is not None:
        x = x.astype(int) | mask_metal.astype(int)  # pyright: ignore
    return x


def shrink_down(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
) -> np.ndarray:
    x = 1 - x
    dim = (1, x.shape[1])
    noise = noise_obj.get_noise(x.shape).astype(int)
    delta = np.vstack([x[1:, :], np.ones(dim, dtype=int)]).astype(int)
    x = 1 - ((noise & delta) | x.astype(int))
    if mask_cut is not None:
        x = apply_mask_cut(x, mask_cut=mask_cut)
    if mask_metal is not None:
        x = x.astype(int) | mask_metal.astype(int)  # pyright: ignore
    return x


def shrink_up(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
) -> np.ndarray:
    x = 1 - x
    dim = (1, x.shape[1])
    noise = noise_obj.get_noise(x.shape).astype(int)
    delta = np.vstack([np.ones(dim, dtype=int), x[:-1, :]]).astype(int)
    x = 1 - ((noise & delta) | x.astype(int))
    if mask_cut is not None:
        x = apply_mask_cut(x, mask_cut=mask_cut)
    if mask_metal is not None:
        x = x.astype(int) | mask_metal.astype(int)  # pyright: ignore
    return x


def shrink_left(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
) -> np.ndarray:
    x = 1 - x
    dim = (x.shape[0], 1)
    noise = noise_obj.get_noise(x.shape).astype(int)
    delta = np.hstack([np.zeros(dim, dtype=int), x[:, :-1]]).astype(int)
    x = 1 - ((noise & delta) | x.astype(int))
    if mask_cut is not None:
        x = apply_mask_cut(x, mask_cut=mask_cut)
    if mask_metal is not None:
        x = x.astype(int) | mask_metal.astype(int)  # pyright: ignore
    return x


def shrink_right(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
) -> np.ndarray:
    x = 1 - x
    dim = (x.shape[0], 1)
    noise = noise_obj.get_noise(x.shape).astype(int)
    delta = np.hstack([x[:, 1:], np.zeros(dim, dtype=int)]).astype(int)
    x = 1 - ((noise & delta) | x.astype(int))
    if mask_cut is not None:
        x = apply_mask_cut(x, mask_cut=mask_cut)
    if mask_metal is not None:
        x = x.astype(int) | mask_metal.astype(int)  # pyright: ignore
    return x


def add_one(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
) -> np.ndarray:
    x = x.copy()
    i_row, i_clmn = random.choice(np.argwhere(x == 0))

    def form1(x):
        x[(i_row - 1, i_row), (i_clmn, i_clmn - 1)] = 1
        x[(i_row - 1, i_row), (i_clmn - 1, i_clmn - 2)] = 1

    def form2(x):
        x[(i_row - 1, i_row), (i_clmn, i_clmn - 1)] = 1
        x[(i_row - 1, i_row - 2), (i_clmn - 1, i_clmn)] = 1

    def form3(x):
        x[(i_row, i_row - 1), (i_clmn, i_clmn - 1)] = 1
        x[(i_row, i_row - 1), (i_clmn - 1, i_clmn - 2)] = 1

    def form4(x):
        x[(i_row, i_row - 1), (i_clmn, i_clmn - 1)] = 1
        x[(i_row - 1, i_row - 2), (i_clmn, i_clmn - 1)] = 1

    random.choice([form1, form2, form3, form4])(x)

    if mask_cut is not None:
        x = apply_mask_cut(x, mask_cut=mask_cut)
    if mask_metal is not None:
        x = x.astype(int) | mask_metal.astype(int)  # pyright: ignore
    return x


def add_zero(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
) -> np.ndarray:
    x = x.copy()
    i_row, i_clmn = random.choice(np.argwhere(x == 1))

    def form1(x):
        x[(i_row - 1, i_row), (i_clmn, i_clmn - 1)] = 0
        x[(i_row - 1, i_row), (i_clmn - 1, i_clmn - 2)] = 0

    def form2(x):
        x[(i_row - 1, i_row), (i_clmn, i_clmn - 1)] = 0
        x[(i_row - 1, i_row - 2), (i_clmn - 1, i_clmn)] = 0

    def form3(x):
        x[(i_row, i_row - 1), (i_clmn, i_clmn - 1)] = 0
        x[(i_row, i_row - 1), (i_clmn - 1, i_clmn - 2)] = 0

    def form4(x):
        x[(i_row, i_row - 1), (i_clmn, i_clmn - 1)] = 0
        x[(i_row - 1, i_row - 2), (i_clmn, i_clmn - 1)] = 0

    random.choice([form1, form2, form3, form4])(x)

    if mask_cut is not None:
        x = apply_mask_cut(x, mask_cut=mask_cut)
    if mask_metal is not None:
        x = x.astype(int) | mask_metal.astype(int)  # pyright: ignore
    return x


def recursive_add_one(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
    times=1,
) -> np.ndarray:
    if times > 0:
        x = add_one(
            x,
            noise_obj=noise_obj,
            mask_cut=mask_cut,
            mask_metal=mask_metal,
        )
        return recursive_add_one(
            x,
            noise_obj=noise_obj,
            times=times - 1,
            mask_cut=mask_cut,
            mask_metal=mask_metal,
        )
    else:
        return x


def recursive_add_zero(
    x: np.ndarray,
    noise_obj,
    mask_cut: None | np.ndarray = None,
    mask_metal: None | np.ndarray = None,
    times=1,
) -> np.ndarray:
    if times > 0:
        x = add_zero(
            x,
            noise_obj=noise_obj,
            mask_cut=mask_cut,
            mask_metal=mask_metal,
        )
        return recursive_add_zero(
            x,
            noise_obj=noise_obj,
            times=times - 1,
            mask_cut=mask_cut,
            mask_metal=mask_metal,
        )
    else:
        return x


def recursive_(x, noise_obj, func, times=2):
    if times > 0:
        x = func(x, noise_obj)
        return recursive_(x, noise_obj, func, times - 1)
    else:
        return x


def apply_mask_cut(x, mask_cut):
    new = 1 - (x.astype(int) | mask_cut.astype(int))
    new = 1 - (new | mask_cut.astype(int))
    return new


class Noise:
    def __init__(
        self,
        perlin_period,
        noise_type="fractal",
        octaves=3,
        lacunarity=2,
        threshold=0.6,
    ):
        self.perlin_period = perlin_period

        self.noise_type = noise_type
        self.threshold = threshold
        self.octaves = octaves
        self.lacunarity = lacunarity

        self.get_noise = self.init_noise_function(noise_type)

    def init_noise_function(self, noise_type):

        if noise_type == "fractal":

            def fractal_noise(
                shape,
                period=self.perlin_period,
                octaves=self.octaves,
                lacunarity=self.lacunarity,
                threshold=self.threshold,
            ):
                """
                The function generate_fractal_noise_2d combines several octaves of 2D perlin noise to make 2D fractal noise. Its parameters are:

                - shape: shape of the generated array (tuple of 2 ints)
                - period: number of periods of noise to generate along each axis (tuple of 2 ints)
                - octaves: number of octaves in the noise (int)
                - persistence: scaling factor between two octaves (float)
                - lacunarity: frequency factor between two octaves (float)
                - tileable: if the noise should be tileable along each axis (tuple of 2 bools)
                Note: **shape must be a multiple of lacunarity^(octaves-1)*period **
                """
                arr = generate_fractal_noise_2d(
                    shape=shape, res=period, octaves=octaves, lacunarity=lacunarity
                )
                # Map values to the range 0 to 1
                arr = (arr + 1) / 2
                # Round values based on the threshold
                arr = np.where(arr >= threshold, 1, 0)
                return arr

            return fractal_noise

        elif noise_type == "perlin":

            def perlin_noise(
                shape, period=self.perlin_period, threshold=self.threshold
            ):
                """
                The function generate_perlin_noise_2d generates a 2D texture of perlin noise. Its parameters are:

                - shape: shape of the generated array (tuple of 2 ints)
                - period: number of periods of noise to generate along each axis (tuple of 2 ints)
                - tileable: if the noise should be tileable along each axis (tuple of 2 bools)
                Note: shape must be a multiple of period
                """
                arr = generate_perlin_noise_2d(shape=shape, res=period)
                # Map values to the range 0 to 1
                arr = (arr + 1) / 2
                # Round values based on the threshold
                arr = np.where(arr >= threshold, 1, 0)
                return arr

            return perlin_noise
        else:
            assert noise_type == "perlin"
