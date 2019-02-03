# mf-asc
============================================

Juyter notebook common_pipeline.ipynb contains the testing procedure. 
In order to run it, one should perpare a json file with preprocessed dataset that has a specific format:

* [domain]-Q.json - list of corresponding to each test tuples: first element is a list of names of target objects, second element is a test name/desription.
	For example one of the elements of Q is [['7975F87B', '7B7FD834', '769E9CCC', '7C1854CB', '7B94ACCA', '784B9099', '786A0711', '7C004744'], 'all papers from field Anomaly detection']. In current version it can actually be a list of arbitrary information about tests.
* [domain]_preprocessed_data.json contains the following fields:
	* vertex_to_code - dict that maps object names to their indices in similarity matrix, scores and embedding
	* embedding - array of objects embeddings into a latent semantic space
* [domain]_hfs.json - list of corresponding to each test lists of high-fidelity scores for all objects.
* [domain]_V_sims.npy - matrix of pairwise similarities between objects


============================================

Data that was used for experiments in paper Bayesaian Multi-Fidelity Active Search with Co-kriging is stored in this archive: https://www.dropbox.com/s/6ssuufj99082wri/data_for_reproduction.zip?dl=0
