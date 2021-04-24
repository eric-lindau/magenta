import magenta
import note_seq
from magenta.models.music_vae.data import OneHotMelodyConverter
from note_seq.protobuf import music_pb2
import tensorflow
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import pretty_midi
import numpy as np

# idk if this will work properly


def batch_recon_loss(config_name, checkpoint_name, batch_size=4, num_steps=8, seqs):
    music_vae = TrainedModel(
        configs.CONFIG_MAP[config_name],
        batch_size=batch_size,
        checkpoint_dir_or_path=checkpoint_name)

    converter = OneHotMelodyConverter()
    total_diff = 0
    n = 0

    for in_seq in seqs:
        try:
            out_seq = music_vae.interpolate(
                in_seq, in_seq, num_steps=num_steps)
            in_tensor = converter.to_tensors(in_seq)[0][0].astype(int)
            out_tensor = converter.to_tensors(out_seq)[0][0].astype(int)
            diff_arr = np.sum(np.absolute(in_tensor-out_tensor), axis=1)
            n_diff = np.count_nonzero(diff_arr)
            total_diff += n_diff / diff_arr.size
            n += 1
        except:
            print("reconstruction failed for file " + path)

    return total_diff / n


def recon_loss(seq1, seq2):
    in_tensor = converter.to_tensors(seq1)[0][0].astype(int)
    out_tensor = converter.to_tensors(seq2)[0][0].astype(int)
    assert in_tensor.shape == out_tensor.shape
    diff_arr = np.sum(np.absolute(in_tensor-out_tensor), axis=1)
    n_diff = np.count_nonzero(diff_arr)
    return n_diff / diff_arr.size
