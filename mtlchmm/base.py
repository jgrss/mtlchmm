from .errors import logger
from .model import ModelHMM


class MTLCHMM(ModelHMM):

    """A class to load and prepare data"""

    def __init__(self,
                 method='forward-backward',
                 transition_prior=0.1,
                 n_jobs=1,
                 block_size=2000,
                 assign_class=False,
                 class_list=None,
                 out_dir=None,
                 **kwargs):

        """
        Args:
            method (Optional[str]): The method to model. Choices are ['forward-backward'].
            transition_prior (Optional[float or 2d array]): The state transition probabilities for
                class transition from one year to the next. Default is 0.1.

                *If `transition_prior` is a float, the same transition probability applies to each class. If
                    `transition_prior` is a 2d array, the class transitions are treated separately.

            n_jobs (Optional[int]): The number of parallel jobs. Default is 1.
            block_size (Optional[int]): The block size for in-memory processing. Default is 2000.
            assign_class (Optional[bool]): Whether to assign the class value with the maximum probability.
                Default is False.
            class_list (Optional[int list]): When `assign_class`=True, a list of class values to assign
                to max probabilities. Default is None, or assign ordered indices.
            out_dir (Optional[str]): The output directory. Default is None, or write to input directory.
            kwargs (Optional): Keyword arguments for `mpglue` `create_raster`.

        Examples:
            >>> from mtlchmm import MTLCHMM
            >>>
            >>> model = MTLCHMM(transition_prior=0.1,
            >>>                 n_jobs=-1)
            >>>
            >>> model = MTLCHMM(transition_prior=np.array([[]], dtype='float32'),
            >>>                 n_jobs=-1,
            >>>                 assign_class=True,
            >>>                 class_list=[1, 10, 20])
        """

        if method != 'forward-backward':

            logger.error('  The method must be forward-backward.')
            raise NameError

        self.method = method
        self.transition_prior = transition_prior
        self.n_jobs = int(n_jobs)
        self.block_size = int(block_size)
        self.assign_class = assign_class
        self.class_list = class_list
        self.out_dir = out_dir
        self.kwargs = kwargs

        self.lc_probabilities = None
        self.n_steps = None
        self.n_labels = None
        self.rows = None
        self.cols = None
        self.methods = None
        self.image_infos = None
