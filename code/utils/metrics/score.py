# coding: utf-8

__maintainer__ = 'Lrakotoson'
__all__ = ['f1', 'Metrics']

# In[1]:


from tensorflow.keras import backend as K, callbacks
from sklearn.metrics import f1_score


# In[2]:


def recall(y, y_pred):
    tp = K.sum(K.round(K.clip(y * y_pred, 0, 1)))
    pos_p = K.sum(K.round(K.clip(y, 0, 1)))
    return tp/(pos_p + K.epsilon())


# In[3]:


def precision(y, y_pred):
    tp = K.sum(K.round(K.clip(y * y_pred, 0, 1)))
    pred_p = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return tp/(pred_p + K.epsilon())


# In[4]:


def f1(y, y_pred):
    """
    Compute f1 score
    Use:
    >>> from utils.metrics import f1
    >>> model.compile(..., metrics=[f1])
    """
    r_score = recall(y, y_pred)
    p_score = precision(y, y_pred)
    f1_score = 2*((r_score*p_score)/(r_score+p_score+K.epsilon()))


# In[5]:


class Metrics(callbacks.Callback):
    """
    Callback function.
    Use:
    >>> from utils.metrics import Metrics
    >>> metrics = Metrics()
    >>> model.fit(..., callbacks=[metrics])
    """
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        self.f1s=f1_score(targ, predict)
        return

