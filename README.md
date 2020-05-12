# Twitter-Depression-Analysis:

We are using an LSTM with Convolutional neural network to  identify whether a certain tweet reveals depressing language to serve as foundation for better understanding of the correlation between depression and linguistic style.

##Required Libraries for tweets depression Analysis:
keras - a high-level neural networks API running on top of TensorFlow
matplotlib - a Python 2D plotting library which produces publication quality figures
nltk - Natural Language Toolkit
numpy - the fundamental package for scientific computing with Python
pandas - provides easy-to-use data structures and data analysis tools for Python
sklearn - a software machine learning library
tensorflow - an open source machine learning framework for everyone

##DataSet for tweets depression Analysis:
1. download the pretrained vectors for the Word2vec from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit and place it in the directory.
2. make sure to have the required data set and place them in the input directory, Kaggle dataset can be found: https://www.kaggle.com/ywang311/twitter-sentiment/data. 
3. depressive_tweets_processed.txt and depressed.csv should also be placed in the input directory

##Steps tp predict run tweets depression Analysis:
1. in order to scrap tweets through TWITNT run twintScrap.py . 
2. After having all the required libraries run depressionAnalysis.ipynb which contains some preictions examples. 
3. Saved model is given in the saved_model folder. 
