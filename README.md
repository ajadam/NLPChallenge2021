# NLPChallenge2021
AI Challenge hosted by INSA-Toulouse


## TLDR :

* Ensemble of RoBERTa-large
* No preprocessing
* WeTried to use Beyond-Back Translation & GenderSwap methods in order to tackle the fairness problem.

## Validation strategy :
We did not do any cross-validation, because we quickly noticed that a simple hold-out sample was enough to evaluate our models, because the score on the public LB was very close to the one estimated with our hold-out strategy.


## Insights :

We use ideas from the wining team of [Jigsaw multilingual toxic comment classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160862) in order to mitigate the variability. What we do is a soft-voting classifier of multiple RoBERTa-large predictions, the models only differs in the random seed used for initialization of the last layer. In addition, we also make a prediction after each epoch, and make a soft voting classifier of all the epochs. This reduce the need for CV or even having a hold-out set during the training.