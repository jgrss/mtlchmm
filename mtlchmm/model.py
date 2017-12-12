"""
Code source:
    @author: S. Parker Abercrombie
"""

from __future__ import division

import os
import multiprocessing as multi
import ctypes

from .errors import logger
from mpglue import raster_tools

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


def forward_backward(n_sample):

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

    # TODO: implement a user mask
    # lw_mask = time_series[0]
    # if lw_mask == 1:
    #     WATER_PROB_VECTOR

    time_series = d_stack[n_sample::n_samples].reshape(n_steps, n_labels)

    # Compute forward messages
    forward[0, :] = time_series[0, :]

    for t in range(1, n_steps):

        v = np.multiply(time_series[t, :], transition_matrix.T.dot(forward[t-1, :]))
        forward[t, :] = normalize(v)

    # Compute backward messages
    backward[n_steps-1, :] = label_ones

    for t in range(n_steps-1, 0, -1):

        v = np.dot(transition_matrix, np.multiply(time_series[t, :], backward[t, :]))
        backward[t-1, :] = normalize(v)

    belief = np.multiply(forward, backward)
    Z = belief.sum(axis=1)

    # Ignore zero entries
    Z[Z == 0] = 1.

    # Normalize
    belief /= Z.reshape((n_steps, 1))

    # Return belief as flattened vector
    # d_stack[n_sample::n_samples] = belief.ravel()

    return belief.ravel()


def viterbi():

    """
    Use the Viterbi algorithm to determine the most likely series
    of states from a time series.
    """

    return


class ModelHMM(object):

    """A class for Hidden Markov Models"""

    def fit(self, method='forward-backward', transition_prior=.1, n_jobs=1):

        """
        Fits a Hidden Markov Model

        Args:
            method (Optional[str]): The method to model. Choices are ['forward-backward', 'viterbi'].
            transition_prior (Optional[float]): The prior probability for class transition from one year to the next.
            n_jobs (Optional[int]): The number of parallel jobs. Default is 1.
        """

        if MKL_INSTALLED:
            n_threads_ = mkl_rt.MKL_Set_Num_Threads(n_jobs)

        self.method = method
        self.transition_prior = float(transition_prior)
        self.n_jobs = n_jobs
        self.blocks = 1024

        if not hasattr(self, 'lc_probabilities'):
            logger.error('The `fit` method cannot be executed without data.')

        # Setup the transition matrix.
        self._transition_matrix()

        self.methods = {'forward-backward': forward_backward,
                        'viterbi': viterbi}

        # Open the images.
        self.image_infos = [raster_tools.ropen(image) for image in self.lc_probabilities]

        self._setup_out_infos()

        # Iterate over the image block by block.
        self._block_func()

    def _setup_out_infos(self):

        """Creates the output image informations objects"""

        self.o_infos = list()

        for image_info in self.image_infos:

            d_name, f_name = os.path.split(image_info.file_name)
            f_base, f_ext = os.path.splitext(f_name)

            out_name = os.path.join(d_name, '{}_hmm{}'.format(f_base, f_ext))

            if os.path.isfile(out_name):
                os.remove(out_name)

            if os.path.isfile(out_name + '.ovr'):
                os.remove(out_name + '.ovr')

            if os.path.isfile(out_name + '.aux.xml'):
                os.remove(out_name + '.aux.xml')

            self.o_infos.append(raster_tools.create_raster(out_name, image_info))

    def _block_func(self):

        global d_stack, forward, backward, label_ones, n_samples, n_steps, n_labels

        n_steps = self.n_steps
        n_labels = self.n_labels

        if self.method == 'forward-backward':

            forward = np.empty((self.n_steps, self.n_labels), dtype='float32')
            backward = np.empty((self.n_steps, self.n_labels), dtype='float32')

            label_ones = np.ones(self.n_labels, dtype='float32')

        for i in range(0, self.rows, self.blocks):

            n_rows = raster_tools.n_rows_cols(i, self.blocks, self.rows)

            for j in range(0, self.cols, self.blocks):

                n_cols = raster_tools.n_rows_cols(j, self.blocks, self.cols)

                # Total samples in the block.
                n_samples = n_rows * n_cols

                # Setup the block stack.
                # time steps x class layers x rows x columns
                d_stack = np.empty((self.n_steps, self.n_labels, n_rows, n_cols), dtype='float32')

                block_max = 0

                # Load the block stack.
                #   *all time steps + all probability layers @ 1 pixel = d_stack[:, :, 0, 0]
                for step in range(0, self.n_steps):

                    step_array = self.image_infos[step].read(bands2open=self.n_jobs,
                                                             i=i,
                                                             j=j,
                                                             rows=n_rows,
                                                             cols=n_cols)

                    block_max = max(block_max, step_array.max())

                    d_stack[step] = step_array

                import pdb
                pdb.set_trace()

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
                pool = multi.Pool(processes=self.n_jobs)
                hmm_results = pool.map(self.methods[self.method], range(0, n_samples))
                pool.close()
                del pool

                # Parallel(n_jobs=self.n_jobs,
                #          max_nbytes=None)(delayed(self.methods[self.method])(n_sample,
                #                                                              n_samples,
                #                                                              self.n_steps,
                #                                                              self.n_labels)
                #                           for n_sample in range(0, n_samples))

                import pdb
                pdb.set_trace()

                np.asarray(hmm_results, dtype='float32').reshape()

                # Reshape the results.
                d_stack = d_stack.reshape(self.n_steps, self.n_labels, n_rows, n_cols)

                # Write the block results to file.

                # Iterate over each time step.
                for step in range(0, self.n_steps):

                    # Get the image for the
                    #   current time step.
                    out_rst = self.o_infos[step]

                    # Get the array for the
                    #   current time step.
                    d_stack_sub = d_stack[step]

                    # Iterate over each probability layer.
                    for layer in range(0, self.n_labels):

                        # Write the block for the
                        #   current probability layer.
                        out_rst.write_array(d_stack_sub[layer],
                                            i=i,
                                            j=j,
                                            band=layer+1)

                        out_rst.close_band()

        self.close()

        out_rst = None

    def close(self):

        for i_info in self.image_infos:

            i_info.close()
            del i_info

        del self.image_infos

        for o_info in self.o_infos:

            o_info.close_file()
            del o_info

        del self.o_infos

    def _transition_matrix(self):

        """
        Constructs the transition matrix

        Attributes:
            transition_matrix
        """

        global transition_matrix

        transition_matrix = np.empty((self.n_labels, self.n_labels), dtype='float32')
        transition_matrix.fill(self.transition_prior)
        np.fill_diagonal(transition_matrix, 1. - self.transition_prior)
