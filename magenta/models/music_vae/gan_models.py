from .base_model import MusicVAE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

ds = tfp.distributions


class LatentDiscriminator():
  def __init__(self, z_size):
    self._cross_entropy_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    self._d = keras.Sequential(
      layers.Dense(z_size, activation='sigmoid'),
      layers.Dense(2048, activation='sigmoid'),
      layers.Dense(2048, activation='sigmoid'),
      layers.Dense(1, activation='sigmoid'),
    )

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
    super(MusicVAE, self).__init__(encoder, decoder)
    self.discriminator = discriminator

  # TODO: add discriminator loss using data point(s) and latent point(s)
  def _compute_model_loss(
      self, input_sequence, output_sequence, sequence_length, control_sequence):
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

    # Either encode to get `z`, or do unconditional, decoder-only.
    if hparams.z_size:  # vae mode:
      q_z = self.encode(input_sequence, x_length, control_sequence)
      z = q_z.sample()

      # Prior distribution.
      p_z = ds.MultivariateNormalDiag(
        loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size)

      # KL Divergence (nats)
      # kl_div = ds.kl_divergence(q_z, p_z)

      # Concatenate the Z vectors to the inputs at each time step.
    else:  # unconditional, decoder-only generation
      # kl_div = tf.zeros([batch_size, 1], dtype=tf.float32)
      z = None

    r_loss, metric_map = self.decoder.reconstruction_loss(
      x_input, x_target, x_length, z, control_sequence)[0:2]

    d_loss = self.discriminator.loss(ds.MultivariateNormalDiag(loc=0, scale_diag=1).sample(), z)

    self.loss = tf.reduce_mean(r_loss) + tf.reduce_mean(d_loss)

    # free_nats = hparams.free_bits * tf.math.log(2.0)
    # kl_cost = tf.maximum(kl_div - free_nats, 0)

    # beta = ((1.0 - tf.pow(hparams.beta_rate, tf.to_float(self.global_step)))
    #         * hparams.max_beta)

    # self.loss = tf.reduce_mean(r_loss) + beta * tf.reduce_mean(kl_cost)

    scalars_to_summarize = {
      'loss': self.loss,
      'losses/r_loss': r_loss,
      'losses/d_loss': d_loss,
      # 'losses/kl_loss': kl_cost,
      # 'losses/kl_bits': kl_div / tf.math.log(2.0),
      # 'losses/kl_beta': beta,
    }
    return metric_map, scalars_to_summarize
