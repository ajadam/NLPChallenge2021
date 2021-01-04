# -*- coding: utf-8 -*-

__all__ = [
    "ElectraClassifier",
    "ElectraClassifierPL"
]

import tensorflow as tf
import transformers as tr
from tensorflow.keras import backend as K, layers
from transformers.modeling_tf_utils import TFSequenceClassificationLoss
from transformers.models.electra.modeling_tf_electra import TFElectraPooler


class ElectraClassifier(tr.TFElectraPreTrainedModel, TFSequenceClassificationLoss):
    """
    Classic classifier w/ transformer layer: Electra
    Using CLS token representation
    Output: array of (batch_size, num_labels)
    """
    _keys_to_ignore_on_load_missing = [r"pooler", r"lm_head"]
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.electra = tr.TFElectraMainLayer(config, name = "electra")
        self.stride = layers.Lambda(lambda x: x[:, 0, :], name = "stride")
        self.classifier = layers.Dense(
            config.num_labels,
            activation = tf.keras.activations.softmax,
            name = "classifier")
    
    def call(self, inputs = None, **kwargs):
        outputs = self.electra(inputs, **kwargs)
        sequences = outputs[0]
        cls_token = self.stride(sequences)
        return self.classifier(cls_token)



class ElectraClassifierPL(tr.TFElectraPreTrainedModel, TFSequenceClassificationLoss):
    """
    Classifier w/ encoder layer: Electra
    Using pooled representation
    Output:  array of (batch_size, num_labels)
    """
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.electra = tr.TFElectraMainLayer(config, name = "electra")
        self.pooler = TFElectraPooler(config, name = "pooler")
        self.classifier = layers.Dense(
            config.num_labels,
            activation = tf.keras.activations.softmax,
            name = "classifier")
        
    def call(self, inputs = None, **kwargs):
        outputs = self.electra(inputs, **kwargs)
        pooled = self.pooler(outputs)
        return self.classifier(pooled)