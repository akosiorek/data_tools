import itertools
import numpy as np
from tensorflow.python.util import nest

from scipy.misc import imresize


def constrain_dims(a, b, DIM):
    ai = 0 if a >= 0 else -a
    d = min(DIM - b, 0)
    bi = b - a + d
    return ai, max(bi, 0)


def convert_img_dtype(imgs, dtype):
    if dtype == np.uint8:
        imgs = (imgs - imgs.min()) / (imgs.max() / 255.)
        imgs = imgs.astype(np.uint8)
    return imgs


class TemplateDataset(object):
    """Creates a dataset of floating templates."""

    def __init__(self, canvas_size, n_timesteps):
        """
        :param canvas_size: tuple of ints, size of the canvas that the templates will be placed in
        :param n_timesteps: int, number of timesteps of a sequence
        """

        super(TemplateDataset, self).__init__()
        self._canvas_size = tuple(canvas_size)
        self.n_timesteps = n_timesteps

    def create(self, coords, templates, dtype=np.uint8):
        n_samples = len(templates)
        canvas = np.zeros((self.n_timesteps, n_samples) + self._canvas_size, dtype=np.float32)

        for i, (tjs, seq_templates) in enumerate(itertools.izip(coords, templates)):
            for tj, template in zip(tjs, seq_templates):
                for t in xrange(len(tj)):
                    self._blend(canvas[t, i], template, tj[t])

        return convert_img_dtype(canvas, dtype)

    def _blend(self, canvas, template, pos):
        """Blends `template` into `canvas` at position given by `pos`

        :param canvas:
        :param template:
        :param pos:
        """

        template_shape = template.shape[:2]
        height, width = canvas.shape[:2]

        pos = np.round(pos)
        y0, x0 = pos
        y1, x1 = pos + template_shape
        y0, x0, y1, x1 = (int(i) for i in (y0, x0, y1, x1))

        yt0, yt1 = constrain_dims(y0, y1, height)
        xt0, xt1 = constrain_dims(x0, x1, width)

        y0, y1 = min(max(y0, 0), height), max(min(y1, height), 0)
        x0, x1 = min(max(x0, 0), width), max(min(x1, width), 0)

        # canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], template[yt0:yt1, xt0:xt1])
        self._blend_slice(canvas, template, (y0, y1, x0, x1), (yt0, yt1, xt0, xt1))

    @staticmethod
    def _blend_slice(canvas, template, dst, src):
        """Merges the slice of `template` given by indices in `src` into the slice of `canvas` given by indices `dst`.

        :param canvas:
        :param template:
        :param dst:
        :param src:
        """
        current = canvas[dst[0]:dst[1], dst[2]:dst[3]]
        target = template[src[0]:src[1], src[2]:src[3]]
        canvas[dst[0]:dst[1], dst[2]:dst[3]] = np.maximum(current, target)


class RandomTemplateDataset(TemplateDataset):
    # TODO: Refactor to use the base class more than just the constructor!

    def __init__(self, canvas_size, n_timesteps, templates, trajectory, batch_size=1, n_templates_per_seq=1):
        super(RandomTemplateDataset, self).__init__(canvas_size, n_timesteps)

        self._template = np.asarray(templates)
        self._trajectory = trajectory
        self.batch_size = batch_size

        n_template = nest.flatten(n_templates_per_seq)
        assert (1 <= len(n_template) <= 2)
        if len(n_template) == 1:
            n_template *= 2
        else:
            assert n_template[0] <= n_template[1]

        self.min_templates_per_seq, self.max_templates_per_seq = n_template

    def get_minibatch(self, output_size=None):
        return self(output_size)

    def __call__(self, output_size=None, return_trajectory=False, with_presence=True):
        n_templets_per_seq = np.random.randint(self.min_templates_per_seq, self.max_templates_per_seq + 1,
                                               self.batch_size)
        template_idx = np.cumsum(np.concatenate((n_templets_per_seq, [1])))

        n_templates = n_templets_per_seq.sum()
        random_idx = np.random.choice(len(self._template), n_templates, True)
        templates = self._template[random_idx]

        canvas = np.zeros((self.n_timesteps, self.batch_size) + self._canvas_size, dtype=np.float32)
        trajectory = self._trajectory.create(self.n_timesteps, n_templates, with_presence=with_presence)
        if with_presence:
            trajectory, presence = trajectory
            per_seq_presence = []

        per_seq_trajectories = []
        for b in xrange(self.batch_size):
            st, ed = template_idx[b], template_idx[b + 1]
            seq_templates = templates[st:ed]
            seq_trajectories = trajectory[:, st:ed]
            for t in xrange(self.n_timesteps):
                for i, template in enumerate(seq_templates):
                    self._blend(canvas[t, b], template, seq_trajectories[t, i])
            per_seq_trajectories.append(seq_trajectories)
            if with_presence:
                per_seq_presence.append(presence[st:ed])

        if output_size is not None:
            output = np.empty((self.n_timesteps, self.batch_size) + tuple(output_size), dtype=np.float32)
            for t in xrange(canvas.shape[0]):
                for b in xrange(canvas.shape[1]):
                    output[t, b] = imresize(canvas[t, b], output_size)
            canvas = output

            if return_trajectory:
                ratio = np.asarray(output_size, dtype=np.float32) / np.asarray(self._canvas_size)
                ratio = ratio[np.newaxis, :]
                downsampled_trajectories = nest.flatten(per_seq_trajectories)
                downsampled_trajectories = map(lambda x: x.astype(np.float32) * ratio, downsampled_trajectories)
                per_seq_trajectories = nest.pack_sequence_as(per_seq_trajectories, downsampled_trajectories)

        if return_trajectory:
            canvas = (canvas, per_seq_trajectories)
            if with_presence:
                canvas = canvas + (per_seq_presence,)
        return canvas