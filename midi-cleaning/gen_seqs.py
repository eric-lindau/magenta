import pandas as pd
import numpy as np
import magenta
import note_seq
import pretty_midi


def gen_seqs_w_genre(n=-1):
    midi_df = pd.read_pickle("./genred_midi_full.pkl")
    arr = []
    n = midi_df.size if n < 1 else n
    for i in range(n):
        try:
            midi = pretty_midi.PrettyMIDI(midi_df["Path"][i])
            n_seq = note_seq.midi_to_note_sequence(midi)
            n_seq.sequence_metadata.genre.append(midi_df["Genre"][i])
            arr.append(n_seq)
        except:
            print("midi file cannot be converted")
    return arr
