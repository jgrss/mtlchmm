from .model import ModelHMM
from .errors import logger

import mpglue as gl


class MTLCHMM(ModelHMM):

    """A class to load and prepare data"""

    def __init__(self, lc_probabilities):

        """
        Args:
            lc_probabilities (str list): A list of image class conditional probabilities. Each image in the list
                should be shaped [layers x rows x columns], where layers are equal to the number of land cover
                classes.
        """

        self.lc_probabilities = lc_probabilities
        self.n_steps = len(self.lc_probabilities)

        # Get image information.
        with gl.ropen(lc_probabilities[0]) as i_info:

            self.n_labels = i_info.bands
            self.rows = i_info.rows
            self.cols = i_info.cols

        del i_info

        if not isinstance(self.n_labels, int):
            logger.error('The number of layers was not properly extracted from the image set.')
            raise TypeError

        if not isinstance(self.rows, int):
            logger.error('The number of rows was not properly extracted from the image set.')
            raise TypeError

        if not isinstance(self.cols, int):
            logger.error('The number of columns was not properly extracted from the image set.')
            raise TypeError
