import numpy as np
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, MaxPool1D, \
                                    MaxPool2D, MaxPool3D, Dense, Flatten, Dropout

from mil.errors.custom_exceptions import DimensionError

class MaskedAttentionWeightNorm(tf.keras.layers.Layer):
    """ Doing softmax with mask """
    def __init__(self, **kwargs):
        super(MaskedAttentionWeightNorm, self).__init__(**kwargs)
        
    def call(self, x, mask):
        x = tf.exp(x) * mask
        soft = x / tf.reduce_sum(x, axis=0)
        soft = tf.transpose(soft, [1,0])
        return tf.expand_dims(soft, axis=1)

class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, d, k, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.d = d
        self.k = k

    def build(self, input_shape):
        self.dense_1 = Dense(self.d, activation='tanh')
        self.dense_2 = Dense(self.k)
        
    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
    
class GatedAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, d, k, **kwargs):
        super(GatedAttentionPooling, self).__init__(**kwargs)
        self.d = d
        self.k = k

    def build(self, input_shape):
        self.dense_v = Dense(self.d, activation='tanh')
        self.dense_u = Dense(self.d, activation='sigmoid')
        self.dense_w = Dense(self.k)
        
    def call(self, x):
        a_v = self.dense_v(x)
        a_u = self.dense_u(x)
        x = self.dense_w(a_v*a_u)
        return x
    
class ConvCustom(tf.keras.layers.Layer):
    """ Custom convolution layers, depending on the number of dimensions of the input,
        the size and number of filters is hardcoded, can be changed """
    def __init__(self, **kwargs):
        super(ConvCustom, self).__init__(**kwargs)
        
    def build(self, input_shape):
        if len(input_shape) == 2:
            self.conv_1 = Dense(100, activation='relu', input_shape=input_shape)
            self.max_pool_1 = Dropout(0.2)
            self.conv_2 = Dense(50, activation='relu')
            self.max_pool_2 = Dropout(0.2)
        elif len(input_shape) == 3:
            self.conv_1 = Conv1D(20, kernel_size=5, activation='relu', input_shape=input_shape)
            self.max_pool_1 = MaxPool1D()
            self.conv_2 = Conv1D(50, kernel_size=5, activation='relu')
            self.max_pool_2 = MaxPool1D()
        elif len(input_shape) == 4:
            self.conv_1 = Conv2D(20, kernel_size=5, activation='relu', input_shape=input_shape)
            self.max_pool_1 = MaxPool2D()
            self.conv_2 = Conv2D(50, kernel_size=5, activation='relu')
            self.max_pool_2 = MaxPool2D()
        elif len(input_shape) == 5:
            self.conv_1 = Conv3D(20, kernel_size=5, activation='relu', input_shape=input_shape)
            self.max_pool_1 = MaxPool3D()
            self.conv_2 = Conv3D(50, kernel_size=5, activation='relu')
            self.max_pool_2 = MaxPool3D()
        else:
            raise DimensionError("Input shape not covered by this model")
        
    def call(self, x):
        x = self.conv_1(x)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.max_pool_2(x)
        return x
    
class InstancesRepresentation(tf.keras.layers.Layer):
    """ Represent each instance with the embedding """
    def __init__(self, l, **kwargs):
        super(InstancesRepresentation, self).__init__(**kwargs)
        self.l = l
        
    def build(self, input_shape):
        self.flatten = Flatten()
        self.dense = Dense(self.l, activation='relu')
        
    def call(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        return x   
        
class Masking(tf.keras.layers.Layer):
    """ Helper for masking padded instances """
    def __init__(self, **kwargs):
        super(Masking, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=1, mask_zero=True)
        
    def call(self, padded_x):
        mask = self.embedding.compute_mask(padded_x)
        mask = tf.cast(tf.reduce_any(mask, np.arange(2, len(mask.shape))), dtype=tf.float32)
        mask = tf.transpose(mask, [1,0])
        return mask
    
class AttentionDeepMil(tf.keras.Model):
    def __init__(self, l=500, d=128, k=1, gated=True,**kwargs):
        super(AttentionDeepMil, self).__init__(**kwargs)
        self.d = d
        self.k = k
        self.l = l
        self.gated = gated
        
    def build(self, input_shape):
        self.conv = ConvCustom()
        self.inst_repr = InstancesRepresentation(l=self.l)
        
        if self.gated:
            self.att = AttentionPooling(d=self.d, k=self.k)
        else:
            self.att = GatedAttentionPooling(d=self.d, k=self.k)
            
        self.att_norm = MaskedAttentionWeightNorm()
        self.dense = Dense(1, activation='sigmoid')
        self.masking = Masking()
        
    def call(self, padded_x):
        mask = self.masking(padded_x)
        
        # reshape to process all instances at once
        feat_shape = padded_x.shape[2:]
        res = [-1]
        for e in feat_shape: res.append(e)
        x = tf.reshape(padded_x, res)
        #x = tf.reshape(padded_x, [-1,28,28,1])
        
        # conv layer
        x = self.conv(x)
        ins_rep = self.inst_repr(x)
        ins_rep = tf.reshape(ins_rep, [-1, padded_x.shape[1], self.l])
                           
        # calculate attention weight norm
        att = self.att(ins_rep)
        att = tf.transpose(tf.squeeze(att, axis=-1), [1,0])
        att = self.att_norm(att, mask)
                           
        # multiply weights w/ instance representation. bag representation
        m = tf.matmul(att, ins_rep)
        m = tf.reshape(m, [-1, self.l])
                           
        # classification
        y = self.dense(m)
        return y, att
    
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            y_pred, _ = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients,  self.trainable_variables))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

class AttentionDeepPoolingMil(KerasClassifier):
    """ 
    Attention Deeep Model using the Keras classifier class, which uses a 
    sklearn like structure.
    
    from paper
    Attention-based Deep Multiple Instance Learning (Maximilian Ilse, Jakub M. Tomczak, Max Welling)
    https://arxiv.org/abs/1802.04712
    """
    def __init__(self, gated=True, threshold=0.2, loss='binary_crossentropy', optimizer='adam'):
        self.gated = gated
        self.loss = loss
        self.optimizer = optimizer
        self.threshold = threshold
        self.model = super(AttentionDeepPoolingMil, self).__init__(build_fn=self.build)
    
    def build(self):
        model = AttentionDeepMil(gated=self.gated)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model
    
    def predict(self, X_test, **kwargs):
        return self.model.predict(X_test, **kwargs)[0]
        
    def get_positive_instances(self, X, **kwargs):
        """ Get instances with greater impact on the bag embedding

        Parameters
        ----------
        X : contains the bags to predict the positive instances
        threshold : value between 0 and 1. If the weighted sum of instances
                    representation has more than threshold value for an instance
                    then the instance is marked as positive.

        Returns
        -------
        pos_ins : a list containing the indexs of the positive instances in X

        """
        
        y_pred, att = self.model.predict(X, **kwargs)
        att = att.reshape(len(X), -1)
        y_pred = y_pred.reshape(len(X), -1)
        pos_ins = tf.where((att > self.threshold) & (y_pred > 0.5))
        return pos_ins