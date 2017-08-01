Multi-temporal land cover maps with a Hidden Markov Model (MTLCHMM)
---

## Reference

> Abercrombie, S Parker and Friedl, Mark A (2016) Improving the Consistency of Multitemporal Land 
Cover Maps Using a Hidden Markov Model. _IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING_, 54(2), 703--713.

### Usage

```python
>>> import mtlchmm
>>>
>>> hmm_model = mtlchmm.MTLCHMM(['/lc_probas_yr01.tif', 
>>>                              '/lc_probas_yr02.tif',
>>>                              '/lc_probas_yr03.tif'])
>>>
>>> hmm_model.fit(method='forward-backward', 
>>>               transition_prior=.1, 
>>>               n_jobs=-1)
```

```text
Results from the above example would be written to:

/lc_probas_yr01_hmm.tif
/lc_probas_yr02_hmm.tif
/lc_probas_yr03_hmm.tif
```