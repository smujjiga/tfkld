## Fast and scalable implemtation of Term Frequencey Kullback Leibler Divergence (TFKLD)

This reimplementation is based on [https://github.com/jiyfeng/tfkld](https://github.com/jiyfeng/tfkld) aimed to drastically speeds up weight calculation.TFKLD was propsed in this [2013 EMNLP paper](http://www.cc.gatech.edu/~jeisenst/papers/ji-emnlp-2013.pdf). 

Also available is the test script ([fe_quora.py](https://github.com/smujjiga/tfkld/blob/master/fe_quora.py)) to extract TFKLD features of the [Quora dataset](https://www.kaggle.com/c/quora-question-pairs) hosted on kaggle as part of a competition. 

## Steps to run the test script
* Download the dataset from [here](https://www.kaggle.com/c/quora-question-pairs/data).
* Extract the zip file and place it in the same directory as that of tflkd.py and fe_quora.py
* Execute fe_quora.py.
* It should take some time and after it finishes you should have train-tfkld-dr.pkl, dev-tfkld-dr.pkl and test-tfkld-dr.pkl pickle files corresponding to the test, development and test data corrspondingly. 
* TFKLD features are reduced to 200 dimensions using SVD. 