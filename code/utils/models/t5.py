# -*- coding: utf-8 -*-

__all__ = [
    "T5Generator"
]

import tensorflow as tf
import transformers as tr
from tensorflow.keras import backend as K


class T5Generator(tr.TFT5ForConditionalGeneration):
    """
    Classifier-Generator with language modeling head on top
    T5 Encoder-Decoder model
    Sequence output
    """

    def __init__(self, *args, log_dir = None, cache_dir = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss') 
    
    def train_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.trainable_variables)
            
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        lr = self.optimizer._decayed_lr(tf.float32)
        
        self.loss_tracker.update_state(loss)        
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({'lr': lr})
        
        return metrics


    def test_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]
        
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


    def batch_generate(self, inputs, batch_size, **kwargs):
        """
        Generates data per batch.
        inputs: tensor of token ids
        batch_size: Size of the data to be generated at a time.
        :**kwargs: generate kwargs
        """
        generated = list()
        size = len(inputs)-1
        for i in range(0, size, batch_size):
            print('\r', end = f'{i}/{size} generated', flush = True)
            y = self.generate(inputs[i:i+batch_size], **kwargs)
            generated.append(y)
        print('\r', end = f'{size}/{size} generated', flush = True)
        tensor = tf.ragged.constant(sum(map(
            lambda x:x.numpy().tolist(),
            generated), []), dtype = tf.int32)
        return tensor
    

    def predict(self, *args):
        raise NotImplementedError("predict not implemented")