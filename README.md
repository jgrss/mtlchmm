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

### Usage

```python
>>> import mtlchmm
>>>
>>> hmm_model = mtlchmm.MTLCHMM(['/lc_probas_yr01.tif', 
>>>                              '/lc_probas_yr02.tif',
>>>                              '/lc_probas_yr03.tif'])
>>>
>>> hmm_model.fit(method='forward-backward', 
>>>               transition_prior=0.1,
>>>               n_jobs=-1)
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
>>> # Sample land cover
>>> cl.split_samples('/samples.txt')
>>>
>>> # Train a Random Forest classification model and
>>> #   return class conditional probabilities.
>>> cl.construct_model(classifier_info={'classifier': 'RF',
>>>                                     'trees': 1000,
>>>                                     'max_depth': 25},
>>>                    get_probs=True)
>>>
>>> # Predict class conditional probabilities and write to file.
>>> lc_probabilities = list()
>>>
>>> for im in ['/yr01.tif', '/yr02.tif', '/yr03.tif']:
>>>
>>>     out_probs = '/lc_probas_{}'.format(im)
>>>
>>>     cl.predict(im, out_probs)
>>>
>>>     lc_probabilities.append(out_probs)
>>>
>>> # Get the class transitional probabilities.
>>> hmm_model = mtlchmm.MTLCHMM(lc_probabilities)
>>>
>>> # Fit the model
>>> hmm_model.fit(method='forward-backward', 
>>>               transition_prior=0.1,
>>>               n_jobs=-1)
```

Installation
---

### Dependencies

* NumPy
* [MpGlue](https://github.com/jgrss/mpglue)

### Clone the latest version

```bash
git clone https://github.com/BU-LCSC/mtlchmm.git
cd mtlchmm
python setup.py install
```

### Updating

```bash
cd mtlchmm
git pull origin master
python setup.py install
```
