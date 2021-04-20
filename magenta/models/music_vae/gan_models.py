from .base_model import MusicVAE

import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

ds = tfp.distributions


class LatentDiscriminator():
  def build(self, z_size):
    self._cross_entropy_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    self._d = keras.Sequential()
    self._d.add(layers.Dense(z_size, activation='sigmoid'))
    self._d.add(layers.Dense(2048, activation='sigmoid'))
    self._d.add(layers.Dense(2048, activation='sigmoid'))
    self._d.add(layers.Dense(1, activation='sigmoid'))

  def loss(self, real, fake):
    # https://www.tensorflow.org/tutorials/generative/dcgan
    real_loss = self._cross_entropy_fn(tf.ones_like(real), real)
    fake_loss = self._cross_entropy_fn(tf.zeros_like(fake), fake)
    total_loss = real_loss + fake_loss
    return total_loss


class AdversarialMusicVAE(MusicVAE):
  """
  Adversarial extension of MusicVAE.
  Extends VAE loss with discriminator loss to better regularize encodings
  """

  def __init__(self, encoder, decoder, discriminator):
    super(AdversarialMusicVAE, self).__init__(encoder, decoder)
    self._discriminator = discriminator

  @property
  def discriminator(self):
    return self._discriminator

  def build(self, hparams, output_depth, is_training):
    """Builds encoder and decoder.

    Must be called within a graph.

    Args:
      hparams: An HParams object containing model hyperparameters. See
          `get_default_hparams` below for required values.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model will be used for training.
    """
    tf.logging.info('Building adversarial MusicVAE model with %s, %s, %s, and hparams:\n%s',
                    self.encoder.__class__.__name__,
                    self.decoder.__class__.__name__,
                    self.discriminator.__class__.__name__,
                    hparams.values())
    super(AdversarialMusicVAE, self).build(hparams, output_depth, is_training)
    self._discriminator.build(hparams.z_size)

  def train(self, input_sequence, output_sequence, sequence_length,
                          control_sequence=None):
    """Train on the given sequences, returning multiple optimizers.

    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
          identical).
      control_sequence: (Optional) sequence on which to condition. This will be
          concatenated depthwise to the model inputs for both encoding and
          decoding.

    Returns:
      optimizer: A tf.train.Optimizer.
    """

    hparams = self.hparams
    lr = ((hparams.learning_rate - hparams.min_learning_rate) *
          tf.pow(hparams.decay_rate, tf.to_float(self.global_step)) +
          hparams.min_learning_rate)

    _, scalars_to_summarize = self._compute_model_loss(
      input_sequence, output_sequence, sequence_length, control_sequence)

    r_optimizer = tf.train.AdamOptimizer(lr)

    tf.summary.scalar('learning_rate_reconstruction', lr)
    for n, t in scalars_to_summarize.items():
      tf.summary.scalar(n, tf.reduce_mean(t))

    _, scalars_to_summarize = self._compute_discriminator_loss(
      input_sequence, output_sequence, sequence_length, control_sequence)

    d_optimizer = tf.train.AdamOptimizer(lr)

    tf.summary.scalar('learning_rate_discriminator', lr)
    for n, t in scalars_to_summarize.items():
      tf.summary.scalar(n, tf.reduce_mean(t))

    return [r_optimizer, d_optimizer]

  @property
  def losses(self):
    return [self.r_loss, self.d_loss]

  def _compute_model_loss(self, input_sequence, output_sequence, sequence_length, control_sequence):
    """Builds a model with loss for train/eval."""
    hparams = self.hparams
    batch_size = hparams.batch_size

    input_sequence = tf.to_float(input_sequence)
    output_sequence = tf.to_float(output_sequence)

    max_seq_len = tf.minimum(tf.shape(output_sequence)[1], hparams.max_seq_len)

    input_sequence = input_sequence[:, :max_seq_len]

    if control_sequence is not None:
      control_depth = control_sequence.shape[-1]
      control_sequence = tf.to_float(control_sequence)
      control_sequence = control_sequence[:, :max_seq_len]
      # Shouldn't be necessary, but the slice loses shape information when
      # control depth is zero.
      control_sequence.set_shape([batch_size, None, control_depth])

    # The target/expected outputs.
    x_target = output_sequence[:, :max_seq_len]
    # Inputs to be fed to decoder, including zero padding for the initial input.
    x_input = tf.pad(output_sequence[:, :max_seq_len - 1],
                     [(0, 0), (1, 0), (0, 0)])
    x_length = tf.minimum(sequence_length, max_seq_len)

    q_z = self.encode(input_sequence, x_length, control_sequence)
    z = q_z.sample()

    r_loss, metric_map = self.decoder.reconstruction_loss(
      x_input, x_target, x_length, z, control_sequence)[0:2]

    self.r_loss = tf.reduce_mean(r_loss)

    scalars_to_summarize = {
      'r_loss': self.r_loss,
    }
    return metric_map, scalars_to_summarize

  def _compute_discriminator_loss(self, input_sequence, output_sequence, sequence_length, control_sequence):
    hparams = self.hparams
    batch_size = hparams.batch_size

    input_sequence = tf.to_float(input_sequence)
    output_sequence = tf.to_float(output_sequence)

    max_seq_len = tf.minimum(tf.shape(output_sequence)[1], hparams.max_seq_len)

    input_sequence = input_sequence[:, :max_seq_len]

    if control_sequence is not None:
      control_depth = control_sequence.shape[-1]
      control_sequence = tf.to_float(control_sequence)
      control_sequence = control_sequence[:, :max_seq_len]
      # Shouldn't be necessary, but the slice loses shape information when
      # control depth is zero.
      control_sequence.set_shape([batch_size, None, control_depth])

    # The target/expected outputs.
    x_target = output_sequence[:, :max_seq_len]
    # Inputs to be fed to decoder, including zero padding for the initial input.
    x_input = tf.pad(output_sequence[:, :max_seq_len - 1],
                     [(0, 0), (1, 0), (0, 0)])
    x_length = tf.minimum(sequence_length, max_seq_len)

    # Prior distribution.
    p_z = ds.MultivariateNormalDiag(
      loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size)

    q_z = self.encode(input_sequence, x_length, control_sequence)
    z = q_z.sample()

    # TODO: ensure the shape is right. it should be (batch_size, 1)... I think
    d_loss = self._discriminator.loss(p_z.sample(), z)

    self.d_loss = tf.reduce_mean(d_loss)

    scalars_to_summarize = {
      'd_loss': self.d_loss,
    }
    return {}, scalars_to_summarize
