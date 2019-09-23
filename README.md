[](#mit-license)[](#python-3.6)[](#package-version)

[![MIT license](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-black.svg)](https://www.python.org/downloads/release/python-360/)
![Package version](https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000)

Multi-temporal land cover maps with a Hidden Markov Model (MTLCHMM)
---

This is largely a port of S. Parker Abercrombie's code. The main differences between the 
original code and this library are:

1. MTLCHMM is designed to support any sensor.
2. Parallel processing at the pixel level is handled differently (however, no tests 
between the two have been conducted).

## Reference

> Abercrombie, S Parker and Friedl, Mark A (2016) [Improving the Consistency of Multitemporal Land
Cover Maps Using a Hidden Markov Model](https://ieeexplore.ieee.org/document/7254169/). _IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING_, 54(2), 703--713.

## New in `v. 0.1.0`

* Image I/O is now handled with [`Rasterio`](https://rasterio.readthedocs.io/en/stable/).

### Usage

```python
>>> import mtlchmm
>>>
>>> hmm_model = mtlchmm.MTLCHMM(method='forward-backward', 
>>>                             transition_prior=0.1,
>>>                             block_size=512,
>>>                             n_jobs=-1,
>>>                             tiled=True,
>>>                             blockxsize=512,
>>>                             blockysize=512,
>>>                             compress='lzw')
>>>
>>> hmm_model.fit_predict(['lc_probas_yr01.tif', 
>>>                        'lc_probas_yr02.tif',
>>>                        'lc_probas_yr03.tif'])
```

```text
Results from the above example would be written to:

/lc_probas_yr01_hmm.tif
/lc_probas_yr02_hmm.tif
/lc_probas_yr03_hmm.tif
```

### Full example with classification using [MpGlue](https://github.com/jgrss/mpglue)

```python
>>> import mpglue as gl
>>> import mtlchmm
>>>
>>> cl = gl.classification()
>>>
>>> # Load land cover samples.
>>> cl.split_samples('/samples.txt')
>>>
>>> # Train a Random Forest classification model.
>>> cl.construct_model(classifier_info={'classifier': 'RF',
>>>                                     'trees': 1000,
>>>                                     'max_depth': 25})
>>>
>>> lc_probabilities = list()
>>>
>>> # Predict class conditional probabilities and write to file.
>>> for im in ['yr01.tif', 'yr02.tif', 'yr03.tif']:
>>>
>>>     out_probs = '{}_probas.tif'.format(im.split('.')[0])
>>>
>>>     cl.predict(im, 
>>>                out_probs,
>>>                predict_probs=True)
>>>
>>>     # Store output file names.
>>>     lc_probabilities.append(out_probs)
>>>
>>> # Get the class transitional probabilities.
>>> hmm_model = mtlchmm.MTLCHMM(method='forward-backward', 
>>>                             transition_prior=0.1,
>>>                             n_jobs=-1)
>>>
>>> # Fit the HMM model and write 
>>> #   adjusted probabilities to file.
>>> hmm_model.fit_predict(lc_probabilities)
```

Installation
---

### Dependencies

* NumPy
* Rasterio

#### Install from GitHub

```commandline
> pip install git+https://github.com/jgrss/mtlchmm.git
```

#### Alternatively, clone the latest version

```commandline
> git clone https://github.com/jgrss/mtlchmm.git
> cd mtlchmm/
> python setup.py install
```

#### Updating

```commandline
> cd mtlchmm/
> git pull origin master
> python setup.py install
```
