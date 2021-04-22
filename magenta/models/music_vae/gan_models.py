from .base_model import MusicVAE

import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

ds = tfp.distributions


class LabelDiscriminator():
  def build(self, hparams):
    self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    self.D = keras.Sequential()
    self.D.add(layers.Dense(hparams.n_clusters + 1, activation='sigmoid'))
    self.D.add(layers.Dense(2048, activation='sigmoid'))
    self.D.add(layers.Dense(2048, activation='sigmoid'))
    self.D.add(layers.Dense(1, activation='sigmoid'))

  def loss(self, real, fake):
    real_dist = self.D(real)
    fake_dist = self.D(fake)
    real_loss = self.cross_entropy(tf.ones_like(real_dist), real_dist)
    fake_loss = self.cross_entropy(tf.zeros_like(fake_dist), fake_dist)
    total_loss = real_loss + fake_loss
    return total_loss


class LatentDiscriminator():
  def build(self, hparams):
    self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    self.D = keras.Sequential()
    self.D.add(layers.Dense(hparams.z_size, activation='sigmoid'))
    self.D.add(layers.Dense(2048, activation='sigmoid'))
    self.D.add(layers.Dense(2048, activation='sigmoid'))
    self.D.add(layers.Dense(1, activation='sigmoid'))

  def loss(self, real, fake):
    # https://www.tensorflow.org/tutorials/generative/dcgan
    real_dist = self.D(real)
    fake_dist = self.D(fake)
    real_loss = self.cross_entropy(tf.ones_like(real_dist), real_dist)
    fake_loss = self.cross_entropy(tf.zeros_like(fake_dist), fake_dist)
    total_loss = real_loss + fake_loss
    return total_loss


class AdversarialMusicVAE(MusicVAE):
  """
  Adversarial extension of MusicVAE.
  Extends VAE loss with discriminator loss to better regularize encodings
  """

  # TODO: parametrize priors
  def __init__(self, encoder, decoder, D_latent, D_label):
    super(AdversarialMusicVAE, self).__init__(encoder, decoder)
    self.D_latent = D_latent
    self.D_label = D_label

  @property
  def discriminator(self):
    return self.D_latent

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
    # Cannot rely on this because we need to adjust encoder input size
    # super(AdversarialMusicVAE, self).build(hparams, output_depth, is_training)
    self.global_step = tf.train.get_or_create_global_step()
    self._hparams = hparams

    self._mu = layers.Dense(
      hparams.z_size,
      name='encoder/mu',
      kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    self._sigma = layers.Dense(
      hparams.z_size,
      activation=tf.nn.softplus,
      name='encoder/sigma',
      kernel_initializer=tf.random_normal_initializer(stddev=0.001))

    self.D_latent.build(hparams)
    self.D_label.build(hparams)

    self.categorical = layers.Dense(
      hparams.n_clusters,
      activation=tf.nn.softplus,
      name='encoder/categorical'
    )

    self._encoder.build(hparams, is_training)

    # build decoder with assumed larger latent size, really it's just concatenated with label
    hparams.z_size += hparams.n_clusters
    self._decoder.build(hparams, output_depth, is_training)
    hparams.z_size -= hparams.n_clusters

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

    _, scalars_to_summarize = self.loss_D_latent(
      input_sequence, output_sequence, sequence_length, control_sequence)

    d_optimizer = tf.train.AdamOptimizer(lr)

    tf.summary.scalar('learning_rate_discriminator', lr)
    for n, t in scalars_to_summarize.items():
      tf.summary.scalar(n, tf.reduce_mean(t))

    return [r_optimizer, d_optimizer]

  @property
  def losses(self):
    return [self.r_loss, self.d_loss]

  def encode_label(self, input_sequence, sequence_length, control_sequence):
    sequence = tf.to_float(input_sequence)
    if control_sequence is not None:
      control_sequence = tf.to_float(control_sequence)
      sequence = tf.concat([sequence, control_sequence], axis=-1)
    encoder_output = self.encoder.encode(sequence, sequence_length)
    return ds.OneHotCategorical(logits=self.categorical(encoder_output))

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

    q_label = self.encode_label(input_sequence, x_length, control_sequence)
    label = tf.to_float(q_label.sample())

    r_loss, metric_map = self.decoder.reconstruction_loss(
      x_input, x_target, x_length, tf.concat([label, z], -1), control_sequence)[0:2]

    self.r_loss = tf.reduce_mean(r_loss)

    scalars_to_summarize = {
      'r_loss': self.r_loss,
    }
    return metric_map, scalars_to_summarize

  # TODO: mention logits in report. essentially "this one is more likely to be drawn from". all relative
  # TODO: *** the new network outputs parameters to a categorical distribution. regularized by prior
  def loss_D_label(self, input_sequence, output_sequence, sequence_length, control_sequence):
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

    # Prior distribution. Uniform categorical
    p_label = ds.OneHotCategorical(probs=[1 / hparams.n_clusters for _ in range(hparams.n_clusters)])
    uniform_label = tf.stack([p_label.sample() for _ in range(batch_size)])

    q_label = self.encode_label(input_sequence, x_length, control_sequence)
    label = q_label.sample()

    self.d_loss = tf.reduce_mean(self.D_latent.loss(uniform_label, label))

    scalars_to_summarize = {
      'd_loss': self.d_loss,
    }
    return self.d_loss, scalars_to_summarize

  def loss_D_latent(self, input_sequence, output_sequence, sequence_length, control_sequence):
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

    # Prior distribution. Standard gaussian
    p_z = ds.MultivariateNormalDiag(
      loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size)
    gaussian_z = tf.stack([p_z.sample() for _ in range(batch_size)])

    q_z = self.encode(input_sequence, x_length, control_sequence)
    z = q_z.sample()

    self.d_loss = tf.reduce_mean(self.D_latent.loss(gaussian_z, z))

    scalars_to_summarize = {
      'd_loss': self.d_loss,
    }
    return self.d_loss, scalars_to_summarize
