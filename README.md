# NLPChallenge2021
AI Challenge hosted by INSA-Toulouse

## TLDR :

* Ensemble of RoBERTa-large
* No preprocessing (on text)
* WeTried to use Beyond-Back Translation & GenderSwap methods in order to tackle the fairness problem.

## Report
[**Fairness and Job description classification**](/report/master.pdf) (FR)

## [Notebooks](/notebooks/)
- [Preprocessing](/notebooks/Preprocessing.ipynb): How we perform label cleaning, data augmentation and gender swapping before training our models to improve fairness
- [Training](/notebooks/Training.ipynb): How we train our models after preprocessing (RoBERTa, Electra, T5)
- [TPU-RoBERTa](/notebooks/tpu-roberta-large.ipynb): How we specifically trained Large RoBERTa's weights
- [Submission](/notebooks/submission-ensemble.ipynb): Template notebook we used for Assemble.

## Datasets:
- We saved all datasets produced by [`Preprocessing`](/notebooks/Preprocessing.ipynb) notebook [here](https://drive.google.com/drive/folders/1QyPvtM-cVdtwztnyWsLMFQAT8Bejv0nd?usp=sharing)
- The difference between those data are explained in [`Training`](/notebooks/Training.ipynb) notebook (and links for the weights)

## Validation strategy :
We did not do any cross-validation, we quickly noticed that a simple hold-out sample was enough to evaluate our models, because the score on the public LB was very close to the one estimated with our hold-out strategy.

## Insights :

We use ideas from the winning team of [Jigsaw multilingual toxic comment classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160862) in order to mitigate the variability. What we do is a soft-voting classifier of multiple RoBERTa-large predictions, the models only differs in the random seed used for initialization of the last layer. In addition, we also make a prediction after each epoch, and average them for the final prediction. This removed the need to pick a best epoch for predictions, which is pretty big.