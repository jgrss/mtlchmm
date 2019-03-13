"""
Code source:
    @author: S. Parker Abercrombie
"""

from __future__ import division
from builtins import int

import os
from copy import copy
import ctypes

from .errors import logger
from .pool import pooler

from mpglue import raster_tools
from mpglue.stats import _lin_interp

try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

try:
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    MKL_INSTALLED = True
except:
    MKL_INSTALLED = False


def normalize(v):

    """
    Normalizes a probability vector by dividing each element by the
    sum of the elements. The elements are probabilities, which are
    assumed to be in the range [0, 1]. Returns v unmodified if the sum
    is <= 0.0.
    """

    Z = v.sum()

    if Z > 0:
        return v / Z

    return v


def _forward(time_series, fc):

    """
    Forward algorithm
    """

    # Initial probability
    fc[0] = time_series[0]

    for t in range(1, n_steps):
        fc[t] = normalize(time_series[t] * transition_matrix.dot(fc[t-1]))

    return fc


def _backward(time_series, bc):

    # Initial probability
    bc[n_steps-1] = 1.0

    for t in range(n_steps-1, 0, -1):
        bc[t-1, :] = normalize(np.dot(transition_matrix, (time_series[t] * bc[t])))

    return bc


def _likelihood(fc, bc):

    posterior = fc * bc

    z = posterior.sum(axis=1)

    # Ignore zero entries
    z[z == 0] = 1.0

    # Normalize
    return (posterior / z.reshape((n_steps, 1))).T


def forward_backward(n_sample):

    fc = forward.copy()
    bc = backward.copy()

    # Time x Labels
    # [t1_l1, t1_l2, ..., t1_ln]
    # [t2_l1, t2_l2, ..., t2_ln]
    time_series = d_stack[n_sample::n_samples].reshape(n_steps, n_labels)

    time_series = _lin_interp.lin_interp(np.float32(time_series.T), 0.0).T

    if time_series.max() == 0:
        return time_series.T

    fc = _forward(time_series, fc)
    bc = _backward(time_series, bc)

    return _likelihood(fc, bc)


def _forward_backward(n_sample):

    """
    Uses the Forward/Backward algorithm to compute marginal probabilities by
    propagating influence forward along the chain.

    Args:
        n_sample (int)

    time_series (2d array): A 2d array (M x N), where M = time steps and N = class labels.
        Each row represents one time step.

    Reference:
        For background on this algorithm see Section 17.4.2 of
        'Machine Learning: A Probabilistic Perspective' by Kevin Murphy.
    """

    time_series = d_stack[n_sample::n_samples].reshape(n_steps, n_labels)

    if time_series.max() == 0:
        return time_series.T

    # Compute forward messages
    forward[0, :] = time_series[0, :]

    for t in range(1, n_steps):
        forward[t, :] = normalize(np.multiply(time_series[t, :], transition_matrix_t.dot(forward[t-1, :])))

    # Compute backward messages
    backward[n_steps-1, :] = label_ones

    for t in range(n_steps-1, 0, -1):
        backward[t-1, :] = normalize(np.dot(transition_matrix, np.multiply(time_series[t, :], backward[t, :])))

    belief = np.multiply(forward, backward)
    Z = belief.sum(axis=1)

    # Ignore zero entries
    Z[Z == 0] = 1.0

    # Normalize
    belief /= Z.reshape((n_steps, 1))

    # Return belief as flattened vector
    # d_stack[n_sample::n_samples] = belief.ravel()

    return belief.T


# TODO
def viterbi():

    """
    Use the Viterbi algorithm to determine the most likely series
    of states from a time series.
    """

    return


def _get_min_extent(image_list):

    min_left = -1e9
    min_right = 1e9
    min_top = 1e9
    min_bottom = -1e9

    for im in image_list:

        with raster_tools.ropen(im) as src:

            min_left = max(min_left, src.left)
            min_right = min(min_right, src.right)
            min_top = min(min_top, src.top)
            min_bottom = max(min_bottom, src.bottom)

            cell_size = src.cellY
            n_layers = src.bands

        src = None

    if ((min_left < 0) and (min_right < 0)) or ((min_left >= 0) and (min_right >= 0)):
        columns = int(round((abs(min_right) - abs(min_left)) / cell_size))
    else:
        columns = int(round((abs(min_right) + abs(min_left)) / cell_size))

    if ((min_bottom < 0) and (min_top < 0)) or ((min_bottom >= 0) and (min_top >= 0)):
        rows = int(round((abs(min_top) - abs(min_bottom)) / cell_size))
    else:
        rows = int(round((abs(min_top) + abs(min_bottom)) / cell_size))

    return min_left, min_bottom, min_right, min_top, cell_size, n_layers, rows, columns


class ModelHMM(object):

    """A class for Hidden Markov Models"""

    def fit_predict(self, lc_probabilities):

        """
        Fits a Hidden Markov Model

        Args:
            lc_probabilities (str list): A list of image class conditional probabilities. Each image in the list
                should be shaped [layers x rows x columns], where layers are equal to the number of land cover
                classes.
        """

        if not lc_probabilities:

            logger.error('The `fit` method cannot be executed without data.')
            raise ValueError

        if MKL_INSTALLED:
            n_threads_ = mkl_rt.MKL_Set_Num_Threads(self.n_jobs)

        self.lc_probabilities = lc_probabilities
        self.n_steps = len(self.lc_probabilities)

        self.left, self.bottom, self.right, self.top, self.cell_size, self.n_labels, self.rows, self.cols = _get_min_extent(self.lc_probabilities)

        if not isinstance(self.n_labels, int):

            logger.error('The number of layers was not properly extracted from the image set.')
            raise TypeError

        if not isinstance(self.rows, int):

            logger.error('The number of rows was not properly extracted from the image set.')
            raise TypeError

        if not isinstance(self.cols, int):

            logger.error('The number of columns was not properly extracted from the image set.')
            raise TypeError

        # Setup the transition matrix.
        self._transition_matrix()

        self.methods = {'forward-backward': forward_backward,
                        'viterbi': viterbi}

        # Open the images.
        self.image_infos = [raster_tools.ropen(image) for image in self.lc_probabilities]

        self._setup_out_infos(**self.kwargs)

        # Iterate over the image block by block.
        self._block_func()

    def _setup_out_infos(self, **kwargs):

        """
        Creates the output image information objects
        """

        if isinstance(self.out_dir, str):

            if not os.path.isdir(self.out_dir):
                os.makedirs(self.out_dir)

        self.o_infos = list()

        for image_info in self.image_infos:

            d_name, f_name = os.path.split(image_info.file_name)
            f_base, f_ext = os.path.splitext(f_name)

            if isinstance(self.out_dir, str):
                d_name = self.out_dir

            out_name = os.path.join(d_name, '{}_hmm{}'.format(f_base, f_ext))

            if os.path.isfile(out_name + '.ovr'):
                os.remove(out_name + '.ovr')

            if os.path.isfile(out_name + '.aux.xml'):
                os.remove(out_name + '.aux.xml')

            if os.path.isfile(out_name):
                self.o_infos.append(raster_tools.ropen(out_name, open2read=False))
            else:

                o_info = image_info.copy()

                o_info.update_info(rows=self.rows,
                                   cols=self.cols,
                                   left=self.left,
                                   top=self.top,
                                   right=self.right,
                                   bottom=self.bottom)

                if self.assign_class:

                    o_info.update_info(storage='byte',
                                       bands=1)

                self.o_infos.append(raster_tools.create_raster(out_name, o_info, **kwargs))

        self.out_blocks = os.path.join(d_name, 'hmm_BLOCK.txt')

    def _block_func(self):

        global d_stack, forward, backward, label_ones, n_samples, n_steps, n_labels

        n_steps = self.n_steps
        n_labels = self.n_labels

        if self.method == 'forward-backward':

            forward = np.empty((self.n_steps, self.n_labels), dtype='float32')
            backward = np.empty((self.n_steps, self.n_labels), dtype='float32')

            label_ones = np.ones(self.n_labels, dtype='float32')

        top_ = copy(self.top)

        for i in range(0, self.rows, self.block_size):

            n_rows = raster_tools.n_rows_cols(i, self.block_size, self.rows)

            left_ = copy(self.left)

            for j in range(0, self.cols, self.block_size):

                hmm_block_tracker = self.out_blocks.replace('_BLOCK', '{:04d}_{:04d}'.format(i, j))

                if os.path.isfile(hmm_block_tracker):
                    continue

                n_cols = raster_tools.n_rows_cols(j, self.block_size, self.cols)

                # Total samples in the block.
                n_samples = n_rows * n_cols

                # Setup the block stack.
                # time steps x class layers x rows x columns
                d_stack = np.empty((self.n_steps, self.n_labels, n_rows, n_cols), dtype='float32')

                block_max = 0

                # Load the block stack.
                #   *all time steps + all probability layers @ 1 pixel = d_stack[:, :, 0, 0]
                for step in range(0, self.n_steps):

                    step_array = self.image_infos[step].read(bands2open=-1,
                                                             y=top_,
                                                             x=left_,
                                                             rows=n_rows,
                                                             cols=n_cols,
                                                             d_type='float32')

                    # step_array /= step_array.max(axis=0)

                    step_array[np.isnan(step_array) | np.isinf(step_array)] = 0

                    block_max = max(block_max, step_array.max())

                    d_stack[step] = step_array

                if block_max == 0:
                    continue

                d_stack = d_stack.ravel()

                # Process each pixel, getting 1
                #   pixel for all time steps.
                #
                # Reshape data to a NxK matrix,
                #   where N is number of time steps and
                #   K is the number of labels.
                #
                # Therefore, each row represents one time step.
                with pooler(processes=self.n_jobs) as pool:
                    hmm_results = np.array(pool.map(self.methods[self.method], range(0, n_samples)), dtype='float32')

                # hmm_results = np.array(map(self.methods[self.method], range(0, 10)), dtype='float32')

                # Parallel(n_jobs=self.n_jobs,
                #          max_nbytes=None)(delayed(self.methods[self.method])(n_sample,
                #                                                              n_samples,
                #                                                              self.n_steps,
                #                                                              self.n_labels)
                #                           for n_sample in range(0, n_samples))

                hmm_results = hmm_results.T.reshape(self.n_steps,
                                                    self.n_labels,
                                                    n_rows,
                                                    n_cols)

                # Reshape the results.
                # d_stack = d_stack.reshape(self.n_steps, self.n_labels, n_rows, n_cols)

                # Write the block results to file.

                # Iterate over each time step.
                for step in range(0, self.n_steps):

                    # Get the image for the
                    #   current time step.
                    out_rst = self.o_infos[step]

                    # Get the array for the
                    #   current time step.
                    hmm_sub = hmm_results[step]

                    if self.assign_class:

                        probabilities_argmax = hmm_sub.argmax(axis=0)

                        if isinstance(self.class_list, list):

                            predictions = np.zeros(probabilities_argmax.shape, dtype='uint8')

                            for class_index, real_class in enumerate(self.class_list):
                                predictions[probabilities_argmax == class_index] = real_class

                        else:
                            predictions = probabilities_argmax

                        out_rst.write_array(predictions,
                                            i=i,
                                            j=j,
                                            band=1)

                        out_rst.close_band()

                    else:

                        # Iterate over each probability layer.
                        for layer in range(0, self.n_labels):

                            # Write the block for the
                            #   current probability layer.
                            out_rst.write_array(hmm_sub[layer],
                                                i=i,
                                                j=j,
                                                band=layer+1)

                            out_rst.close_band()

                if self.track_blocks:

                    with open(hmm_block_tracker, 'wb') as btxt:
                        btxt.write('complete')

                left_ += (n_cols * self.cell_size)

            top_ -= (n_rows * self.cell_size)

        self.close()

        out_rst = None

    def close(self):

        for i_info in self.image_infos:

            i_info.close()
            i_info = None

        self.image_infos = None

        for o_info in self.o_infos:

            o_info.close_file()
            o_info = None

        self.o_infos = None

    def _transition_matrix(self):

        """
        Constructs the transition matrix
        """

        global transition_matrix, transition_matrix_t

        if isinstance(self.transition_prior, np.ndarray):
            transition_matrix = self.transition_prior
        else:

            transition_matrix = np.empty((self.n_labels, self.n_labels), dtype='float32')
            transition_matrix.fill(self.transition_prior)
            np.fill_diagonal(transition_matrix, 1.0 - self.transition_prior)

        transition_matrix_t = transition_matrix.T
