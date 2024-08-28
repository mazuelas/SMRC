
# Leveraging Supervised and Unsupervised Methods for Minimax Semi-supervised Learning

The provided files implement semi-supervised minimax risk classifiers (SMRCs).

SMRCs can leverage the representation and discriminative capabilities of general supervised and unsupervised learning algorithms while providing performance guarantees. 


### Code

(/code) folder contains the Matlab files required to execute the method:

* main.m script that runs SMRCs with the same settings as those for Table 1 in the paper using the dataset `USPS' that can be found in the folder '/data'
* learn_uns.m function that uses principal component analysis and k-means clustering to find features using unsupervised samples
* learn_sup.m function that uses feedforward neural networks boosted ensembles of trees to find features an a supervised classifier using supervised samples
* compute_phi_uns.m function that obtains values for the feature mapping corresponding with the unsupervised algorithms
* compute_phi_sup.m function that obtains values for the feature mapping corresponding with the supervised algorithms
* fit_SMRC.m function that uses stochastic subgradient algorithm to determine the SMRC parameters
* predict_SMRC.m function that evaluates SMRC classifiers for specific instances


## Test case

File `main.m' runs SMRC and obtains error estimates for one random partition of `USPS' dataset. 
