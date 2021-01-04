# -*- coding: utf-8 -*-

__all__ = [
    "RobertaClassifier",
    "RobertaClassifierOVR"
]

import tensorflow as tf
import transformers as tr
from tensorflow.keras import backend as K, layers
from transformers.modeling_tf_utils import TFSequenceClassificationLoss


class RobertaClassifier(tr.TFRobertaPreTrainedModel, TFSequenceClassificationLoss):
    """
    Classic classifier w/ transformer layer: RoBERTa
    Using CLS token representation
    Output: array of (batch_size, num_labels)
    """
    _keys_to_ignore_on_load_missing = [r"pooler", r"lm_head"]
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.roberta = tr.TFRobertaMainLayer(
            config,
            add_pooling_layer = False,
            name = "roberta")
        self.stride = layers.Lambda(lambda x: x[:, 0, :], name = "stride")
        self.classifier = layers.Dense(
            config.num_labels,
            activation = tf.keras.activations.softmax,
            name = "classifier")
    
    def call(self, inputs = None, **kwargs):
        outputs = self.roberta(inputs, **kwargs)
        sequences = outputs[0]
        cls_token = self.stride(sequences)
        return self.classifier(cls_token)


class RobertaClassifierOVR(tr.TFRobertaPreTrainedModel, TFSequenceClassificationLoss):
    """
    Classic classifier w/ transformer layer: RoBERTa
    Using CLS token representation
    Last layer of num_labels Dense binary for OVR classification
    Output: list of num_labels arrays of (batch_size, 1)
    """
    _keys_to_ignore_on_load_missing = [r"pooler", r"lm_head"]
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.roberta = tr.TFRobertaMainLayer(
            config,
            add_pooling_layer = False,
            name = "roberta")
        self.stride = layers.Lambda(lambda x: x[:, 0, :], name = "stride")
        self.construct()
    
    
    def construct(self):
        for label in range(self.num_labels):
            name = f'classifier_{label}'
            setattr(
                self, name,
                layers.Dense(
                    1, name = name,
                    activation = tf.keras.activations.sigmoid
            ))
    
    def callOVR(self, x):
        predictions = list()
        for label in range(self.num_labels):
            clf = getattr(self, f'classifier_{label}')
            predictions.append(clf(x))
        return predictions
    
    
    def call(self, inputs = None, **kwargs):
        outputs = self.roberta(inputs, **kwargs)
        sequences = outputs[0]
        cls_token = self.stride(sequences)
        return self.callOVR(cls_token)