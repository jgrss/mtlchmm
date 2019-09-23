"""
Code source:
    @author: S. Parker Abercrombie
"""

from __future__ import division
from builtins import int

import os
from copy import copy
import ctypes
import multiprocessing as multi
from collections import namedtuple

from .errors import logger

import numpy as np
import rasterio as rio
from rasterio.windows import Window
from affine import Affine

try:
    MKL_LIB = ctypes.CDLL('libmkl_rt.so')
except:
    MKL_LIB = None


def _normalize(v):

    """
    Normalizes a probability vector by dividing each element by the
    sum of the elements. The elements are probabilities, which are
    assumed to be in the range [0, 1]. Returns v unmodified if the sum
    is <= 0.0.
    """

    z = v.sum()

    if z > 0:
        return v / z

    return v


def _forward(time_series, fc):

    """
    Forward algorithm

    Args:
        time_series (2d array): Steps x classes.
        fc (2d array): Forward probabilities.
    """

    # Initial probability (from predicted class posteriors of the first time step)
    fc[0] = time_series[0]

    for t in range(1, n_steps):
        fc[t] = _normalize(time_series[t] * transition_matrix.dot(fc[t-1]))

    return fc


def _backward(time_series, bc):

    """
    Backward algorithm

    Args:
        time_series (2d array): Steps x classes.
        bc (2d array): Backward probabilities.
    """

    # Initial probability
    # bc[n_steps-1] = 1.0 / bc.shape[1]
    bc[n_steps-1] = time_series[-1]

    for t in range(n_steps-1, 0, -1):
        bc[t-1, :] = _normalize(np.dot(transition_matrix, (time_series[t] * bc[t])))

    return bc


def _likelihood(fc, bc):

    """
    Likelihood function

    Args:
        fc (2d array): Forward probabilities.
        bc (2d array): Backward probabilities.

    Returns:
        Posterior probabilities (2d array)
    """

    posterior = fc * bc

    posterior[posterior == 0] = 0.0001

    z = posterior.sum(axis=1)[:, np.newaxis]

    # Ignore zero entries
    # z[z == 0] = 1.0

    # Normalize and transpose
    #
    # The real shape is [time x labels]. The data is
    #   transposed for reshaping and indexing the full
    #   image array.
    return (posterior / z).T


def forward_backward(n_sample):

    fc = forward.copy()
    bc = backward.copy()

    # Time x Labels
    # [t1_l1, t1_l2, ..., t1_ln]
    # [t2_l1, t2_l2, ..., t2_ln]
    time_series = d_stack[n_sample::n_samples].reshape(n_steps, n_labels)

    # time_series = _lin_interp.lin_interp(np.float32(time_series.T), 0.0)

    if time_series.max() == 0:
        return time_series.T

    fc = _forward(time_series, fc)
    bc = _backward(time_series, bc)

    return _likelihood(fc, bc)


# def _forward_backward(n_sample):
#
#     """
#     Uses the Forward/Backward algorithm to compute marginal probabilities by
#     propagating influence forward along the chain.
#
#     Args:
#         n_sample (int)
#
#     time_series (2d array): A 2d array (M x N), where M = time steps and N = class labels.
#         Each row represents one time step.
#
#     Reference:
#         For background on this algorithm see Section 17.4.2 of
#         'Machine Learning: A Probabilistic Perspective' by Kevin Murphy.
#     """
#
#     time_series = d_stack[n_sample::n_samples].reshape(n_steps, n_labels)
#
#     if time_series.max() == 0:
#         return time_series.T
#
#     # Compute forward messages
#     forward[0, :] = time_series[0, :]
#
#     for t in range(1, n_steps):
#         forward[t, :] = _normalize(np.multiply(time_series[t, :], transition_matrix_t.dot(forward[t-1, :])))
#
#     # Compute backward messages
#     backward[n_steps-1, :] = label_ones
#
#     for t in range(n_steps-1, 0, -1):
#         backward[t-1, :] = _normalize(np.dot(transition_matrix, np.multiply(time_series[t, :], backward[t, :])))
#
#     belief = np.multiply(forward, backward)
#     Z = belief.sum(axis=1)
#
#     # Ignore zero entries
#     Z[Z == 0] = 1.0
#
#     # Normalize
#     belief /= Z.reshape((n_steps, 1))
#
#     # Return belief as flattened vector
#     # d_stack[n_sample::n_samples] = belief.ravel()
#
#     return belief.T


# TODO
def viterbi():

    """
    Use the Viterbi algorithm to determine the most likely series
    of states from a time series.
    """

    return


def n_rows_cols(pixel_index, block_size, rows_cols):

    """
    Adjusts block size for the end of image rows and columns.

    Args:
        pixel_index (int): The current pixel row or column index.
        block_size (int): The image block size.
        rows_cols (int): The total number of rows or columns in the image.

    Returns:
        Adjusted block size as int.
    """

    return block_size if (pixel_index + block_size) < rows_cols else rows_cols - pixel_index


def _get_min_extent(image_list):

    min_left = -1e9
    min_right = 1e9
    min_top = 1e9
    min_bottom = -1e9

    for im in image_list:

        with rio.open(im) as src:

            min_left = max(min_left, src.bounds.left)
            min_right = min(min_right, src.bounds.right)
            min_top = min(min_top, src.bounds.top)
            min_bottom = max(min_bottom, src.bounds.bottom)

            cell_size = src.res[0]
            n_layers = src.count

    if ((min_left < 0) and (min_right < 0)) or ((min_left >= 0) and (min_right >= 0)):
        columns = int(round((abs(min_right) - abs(min_left)) / cell_size))
    else:
        columns = int(round((abs(min_right) + abs(min_left)) / cell_size))

    if ((min_bottom < 0) and (min_top < 0)) or ((min_bottom >= 0) and (min_top >= 0)):
        rows = int(round((abs(min_top) - abs(min_bottom)) / cell_size))
    else:
        rows = int(round((abs(min_top) + abs(min_bottom)) / cell_size))

    rows = abs(rows)
    columns = abs(columns)

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

        if MKL_LIB:
            n_threads_ = MKL_LIB.MKL_Set_Num_Threads(self.n_jobs)

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

        self._create_names()

        # Iterate over the image block by block.
        self._block_func()

    def _create_names(self):

        RInfo = namedtuple('RInfo', 'dtype bands crs transform')

        self.out_names = list()

        if self.assign_class:

            dtype = 'uint8'
            bands = 1

        else:

            dtype = 'float32'

            with rio.open(self.lc_probabilities[0]) as src:
                bands = src.count

        with rio.open(self.lc_probabilities[0]) as src:

            self.name_info = RInfo(dtype=dtype,
                                   bands=bands,
                                   crs=src.crs,
                                   transform=Affine.from_gdal(*src.read_transform()))

        for fn in self.lc_probabilities:

            d_name, f_name = os.path.split(fn)
            f_base, f_ext = os.path.splitext(f_name)

            out_name = os.path.join(self.out_dir, '{}_hmm{}'.format(f_base, f_ext))

            self.out_names.append(out_name)

        self.out_blocks = os.path.join(d_name, 'hmm_BLOCK.txt')

        if not isinstance(self.out_dir, str):
            self.out_dir = d_name

        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

    def _block_func(self):

        global d_stack, forward, backward, label_ones, n_samples, n_steps, n_labels

        n_steps = self.n_steps
        n_labels = self.n_labels

        if self.method == 'forward-backward':

            forward = np.empty((self.n_steps, self.n_labels), dtype='float32')
            backward = np.empty((self.n_steps, self.n_labels), dtype='float32')

            label_ones = np.ones(self.n_labels, dtype='float32')

        for i in range(0, self.rows, self.block_size):

            n_rows = n_rows_cols(i, self.block_size, self.rows)

            for j in range(0, self.cols, self.block_size):

                hmm_block_tracker = self.out_blocks.replace('_BLOCK', '{:04d}_{:04d}'.format(i, j))

                if os.path.isfile(hmm_block_tracker):
                    continue

                n_cols = n_rows_cols(j, self.block_size, self.cols)

                w = Window(col_off=j, row_off=i, width=n_cols, height=n_rows)

                # Total samples in the block.
                n_samples = n_rows * n_cols

                # Setup the block stack.
                # time steps x class layers x rows x columns
                d_stack = np.empty((self.n_steps, self.n_labels, n_rows, n_cols), dtype='float32')

                block_max = 0

                # Load the block stack.
                #   *all time steps + all probability layers @ 1 pixel = d_stack[:, :, 0, 0]
                for step in range(0, self.n_steps):

                    # Read all the bands for the current time step.
                    with rio.open(self.lc_probabilities[step]) as src:

                        step_array = src.read(window=w,
                                              out_dtype='float32')

                    step_array[np.isnan(step_array) | np.isinf(step_array)] = 0

                    if isinstance(self.scale_factor, float):
                        step_array *= self.scale_factor

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
                with multi.Pool(processes=self.n_jobs) as pool:
                    hmm_results = np.array(pool.map(self.methods[self.method], range(0, n_samples)), dtype='float32')

                # Reshape the results.
                hmm_results = hmm_results.T.reshape(self.n_steps,
                                                    self.n_labels,
                                                    n_rows,
                                                    n_cols)

                # Write the block results to file.

                # Iterate over each time step.
                for step in range(0, self.n_steps):

                    with rio.open(self.out_names[step],
                                  mode='r+',
                                  height=self.rows,
                                  width=self.cols,
                                  count=self.name_info.bands,
                                  dtype=self.name_info.dtype,
                                  driver=self.driver,
                                  crs=self.name_info.crs,
                                  transform=self.name_info.transform,
                                  **self.kwargs) as dst:

                        # Get the array for the
                        #   current time step.
                        hmm_sub = hmm_results[step]

                        if self.assign_class:

                            probabilities_argmax = hmm_sub.argmax(axis=0)

                            if isinstance(self.class_list, list) or isinstance(self.class_list, np.ndarray):

                                predictions = np.zeros(probabilities_argmax.shape, dtype='uint8')

                                for class_index in range(0, len(self.class_list)):
                                    predictions[probabilities_argmax == class_index] = self.class_list[class_index]

                            else:
                                predictions = probabilities_argmax

                            predictions[hmm_sub.max(axis=0) == 0] = 0

                            dst.write(predictions,
                                      window=w,
                                      indexes=1)

                        else:

                            # Write the probability layers.
                            dst.write(hmm_sub,
                                      window=w,
                                      indexes=list(range(1, self.n_labels+1)))

                if self.track_blocks:

                    with open(hmm_block_tracker, 'wb') as btxt:
                        btxt.write(b'complete')

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
