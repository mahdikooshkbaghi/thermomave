"""Utilities."""

from typing import Optional
import time
from xmlrpc.client import boolean
import numpy as np
import jax.numpy as jnp
import pandas as pd

# Define built-in alphabets
alphabet_dict = {
    "dna": np.array(["A", "C", "G", "T"]),
    "rna": np.array(["A", "C", "G", "U"]),
    "protein": np.array(
        [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
    ),
    "protein*": np.array(
        [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
            "*",
        ]
    ),
}


def x_to_ohe(x: np.ndarray, alphabet: dict, ravel_seqs=True):
    """
    Convert a sequence array to a one-hot encoded matrix.

    Parameters
    ----------
    x: (np.ndarray)
        (N,) array of input sequences, each of length L

    alphabet: (np.ndarray)
        (C,) array describing the alphabet sequences are drawn from.

    ravel_seqs: (bool)
        Whether to return an (N, L*C) array, as opposed to an (N, L, C) array.

    Returns
    -------
    x_ohe: (np.ndarray)
        Array of one-hot encoded sequences, stored as np.int8 values.
    """
    # Get dimensions
    L = len(x[0])
    N = len(x)
    C = len(alphabet)

    # Shape sequences as array of int8s
    x_arr = np.frombuffer(bytes("".join(x), "utf-8"),
                          np.int8, N * L).reshape([N, L])

    # Create alphabet as array of int8s
    alphabet_arr = np.frombuffer(bytes("".join(alphabet), "utf-8"), np.int8, C)

    # Compute (N,L,C) grid of one-hot encoded values
    x_nlc = (x_arr[:, :, np.newaxis] ==
             alphabet_arr[np.newaxis, np.newaxis, :]).astype(np.int8)

    # Ravel if requested
    if ravel_seqs:
        x_ohe = x_nlc.reshape([N, L * C])
    else:
        x_ohe = x_nlc

    return x_ohe


def x_to_stats(x, alphabet, weights=None, verbose=False):
    """
    Identify the consensus sequence from a sequence alignment.

    Parameters
    ----------
    x: (np.ndarray)
        List of sequences. Sequences must all be the same length.

    alphabet: (np.ndarray)
        Alphabet from which sequences are drawn.

    weights: (None, np.ndarray)
        Weights for each sequence. E.g., count values, or numerical y values.
        If None, a value of 1 will be assumed for each sequence.

    verbose: (bool)
        Whether to print computation time.

    Returns
    -------
    consensus_seq: (str)
        Consensus sequence.
    """
    # Start timer
    start_time = time.time()

    # Get alphabet
    alphabet = alphabet_dict[alphabet]

    # Check weights and set if not provided
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = weights.astype(float)
        check(
            len(weights) == len(x),
            f"len(weights)={len(weights)} does not match len(x)={len(x)}",
        )

    # Do one-hot encoding of sequences
    t = time.time()
    x_nlc = x_to_ohe(x, alphabet, ravel_seqs=False)
    N, L, C = x_nlc.shape

    # Dictionary to hold results
    stats = {}

    # Compute x_ohe
    stats["x_ohe_nonravel"] = x_nlc.astype(np.int8)
    stats["x_ohe"] = x_nlc.reshape([N, L * C]).astype(np.int8)

    # Multiply by weights
    x_nlc = x_nlc.astype(float) * weights[:, np.newaxis, np.newaxis]

    # Compute lc encoding of consensus sequence
    x_sum_lc = x_nlc.sum(axis=0)
    x_sum_lc = x_sum_lc.reshape([L, C])
    x_support_lc = x_sum_lc != 0

    # Set number of sequences
    stats["N"] = N

    # Set sequence length
    stats["L"] = L

    # Set number of characters
    stats["C"] = C

    # Set alphabet
    stats["alphabet"] = alphabet

    # Compute probability matrix
    p_lc = x_sum_lc / x_sum_lc.sum(axis=1)[:, np.newaxis]
    stats["probability_df"] = pd.DataFrame(
        index=range(L), columns=alphabet, data=p_lc)

    # Compute sparsity factor
    stats["sparsity_factor"] = (x_nlc != 0).sum().sum() / x_nlc.size

    # Compute the consensus sequence and corresponding matrix.
    # Adding noise prevents ties
    x_sum_lc += 1e-1 * np.random.rand(*x_sum_lc.shape)
    stats["consensus_seq"] = "".join(
        [alphabet[np.argmax(x_sum_lc[l, :])] for l in range(L)])

    # Compute mask dict
    missing_dict = {}
    for l in range(L):
        if any(~x_support_lc[l, :]):
            missing_dict[l] = "".join(alphabet[~x_support_lc[l, :]])
    stats["missing_char_dict"] = missing_dict

    # Provide feedback if requested
    duration_time = time.time() - start_time
    if verbose:
        print(f"\nStats computation time: {duration_time:.5f} sec.\n")

    return stats


def load_dataset(filename: Optional[str] = None,
                 alphabet: Optional[np.array] = 'protein',
                 verbose: Optional[boolean] = True):
    """
    Load dataset provided in data directory.
    Parameters
    ----------
    filename: (str)
        Path to the data set.
    Returns
    -------
    x: (jnp.array)
        One-hot encoded of the sequences.
    y: (jnp.array)
        Measurements.
    L: (int):
        sequence length.
    C: (int)
        alphabet length.

    """
    # Load the dataset
    data_df = pd.read_csv(filename)
    # Get and report sequence length
    L = len(data_df.loc[0, "x"])
    print(f"Sequence length: {L:d} amino acids")
    # protein alphabet length
    C = 20

    # Get the training dataset (x,y)
    # Convert the sequences to the one-hot encoed array

    x_stats = x_to_stats(data_df["x"], alphabet=alphabet, verbose=True)
    x = x_stats["x_ohe_nonravel"]
    # training data is in log2 format
    y = data_df["y"].values

    if verbose == True:
        # Preview dataset
        print(f"\n Dataset looks like:")
        print(data_df.head())
        print(f'\nDataset consists of {len(data_df)} sequence\n')
    return jnp.array(x), jnp.array(y).reshape(-1, 1), L, C
