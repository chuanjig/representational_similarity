# representational_similarity

Scripts I wrote to analyze neural data using searchlight representational similarity analysis (RSA), for an upcoming publication (Wasserman, Chakroff, Saxe, & Young, submitted). My previous commits on this project were made as [@lypsychlab](https://github.com/lypsychlab/).

For this project we:

* analyzed similarities between neural patterns associated with viewing different kinds of moral violations to discover brain regions containing information about different types of moral content
* performed K-means cluster analysis on weights derived from a principal component analysis of behavioral ratings of the same moral violations, as an attempt to discover whether moral categories could be reconstituted from feature information alone 
* analyzed neural similarities _within_ brain regions that encode category information, to see whether feature information was also encoded  


Check out my non-scientist-friendly explanation of RSA under [about_methods](https://github.com/emily-wasserman/about_methods)! 

## Files: 

* **searchlight_base.m** - basic whole-brain searchlight RSA script 
* **rsa_roi.m** - RSA script for voxels inside a masked ROI
* **searchlight_onesamp.m** - aggregate RSA images across subjects 
* **permute_test_roi.m** - run within-ROI RSA, using permuted matrices 
* **load_corrs.m** - aggregate data from within-ROI RSA 
* **orthogonalize_reg** - orthogonalizes regressors in a matrix w/r/t one another
* **behavioral_PCA.R** - run principal components analysis/K-means on behavioral data 

