import numpy as np

from matplotlib.path import Path
from scipy.ndimage.filters import gaussian_filter

from trajectory import NoisyAccelerationTrajectory
from template import RandomTemplateDataset


def polygon_template(shape, verts, value=1., blur=1.0):
    """Creates a polygon template

    Values inside the polygon are filled with `value`

    :param shape: tuple, template size
    :param verts: int or np.array of shape (-1, 2); if int, then `verts`
        random vertices are generated.
    :param value: float, value used to fill the polygon
    :return:
    """
    ny, nx = shape
    if isinstance(verts, int):

        mid = np.asarray(shape, dtype=np.float32) / 2
        angle_interval = 2 * np.pi / verts
        angles = np.arange(0, 2 * np.pi, angle_interval)
        angles += np.random.uniform(.25 * angle_interval, .75 * angle_interval, size=verts)
        r = sum(mid**2)**.5
        rs = np.random.uniform(0.5 * r, r, size=verts)
        s, c = np.sin(angles), np.cos(angles)
        ys = s * rs + mid[0]
        xs = c * rs + mid[1]

        verts = zip(ys, xs)

    assert len(verts) > 2, 'Need at least 3 vertices to create a polygon.'

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(verts)
    grid = path.contains_points(points, radius=0.)
    grid = grid.reshape((ny, nx)).astype(np.float32)

    if blur:
        grid = gaussian_filter(grid, blur)

    return grid / grid.max() * value


def floating_polygon_dataset(n_timesteps, n_samples, n_shapes_per_seq, n_shapes, max_sides=5, canvas_size=(64, 64),
                             template_size=(28, 28), output_size=None):
    """Creates a dataset of floating polygons.

    :param n_timesteps:
    :param n_samples:
    :param n_shapes_per_seq:
    :param n_shapes:
    :param max_sides:
    :param canvas_size:
    :param template_size:
    :param output_size:
    :return:
    """

    allowed_region = np.asarray(canvas_size) - template_size
    y_bounds = [0, allowed_region[0]]
    x_bounds = [0, allowed_region[1]]
    bounds = [y_bounds, x_bounds]
    trajectory = NoisyAccelerationTrajectory(
        noise_std=.01,
        n_dim=2,
        pos_bounds=bounds,
        max_speed=10,
        max_acc=3,
        bounce=True
    )

    templates = []
    for i in xrange(n_shapes):
        sides = np.random.randint(3, max_sides + 1)
        value = np.random.uniform()
        template = polygon_template(template_size, sides, value, blur=False)
        templates.append(template)

    dataset = RandomTemplateDataset(canvas_size, templates, trajectory, n_timesteps,
                              n_templates_per_seq=n_shapes_per_seq, batch_size=n_samples)

    data = dataset(output_size).astype(np.float32)
    data -= data.min()
    data /= data.max()

    return data


if __name__ == '__main__':
    print polygon_template((10, 10), 5)