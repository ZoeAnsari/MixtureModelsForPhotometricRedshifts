# MixtureModelsForPhotometricRedshifts

## Dependencies

Keras

[scikit-learn](https://scikit-learn.org/stable/)

[MDN](https://github.com/ZoeAnsari/keras-mdn-layer)


## Usage

1.By a given set of photometric features for a source as follows:

'g', 'Err_g', 'r', 'Err_r', 'i',  'Err_i', 'z', 'Err_z', 'extinction_i', 'w1', 'w1_sig','w2', 'w2_sig', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z'

One will be able to estimate its probabilistic redshift by changing the data_predict_read_path and data_predict_predict_path in __init__.py to the corresponnding table’s path.


2.To re-train the method with more robust spectroscopic and deeper photometric samples changed the corresponding “path”s in __init__.py.


## Acknowledgement

I acknowledge Adriano Agnello and Christa Gall for the patient guidance and encouragement they have provided thought out the whole process. It would be impossible to implement this method without their supervision. 

## Reference

[Mixture Models for Photometric Redshifts](https://ui.adsabs.harvard.edu/abs/2020arXiv201007319A/abstract)

## Citation

[Mixture Models for Photometric Redshifts](https://ui.adsabs.harvard.edu/abs/2020arXiv201007319A/abstract)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4687149.svg)](https://doi.org/10.5281/zenodo.4687149)

