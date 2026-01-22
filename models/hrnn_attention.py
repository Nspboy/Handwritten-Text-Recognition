"""
Feature Enhancement Module - Step 5 (Core) of the Pipeline

Handwriting Reconstruction Neural Network (HRNN) + Attention

Enhances at the feature level:
- Character alignment
- Stroke consistency
- Spacing between letters

Uses self-attention to focus on important strokes and an HRNN (GRU/LSTM) to refine
the sequence. Operates on BiLSTM outputs (time, batch, hidden).
"""

from __future__ import absolute_import, division, print_function
from utils.tf_compat import tf


def self_attention_layer(
    inputs,
    num_units: int,
    is_training: bool,
    dropout: float = 0.1,
    scope: str = "self_attention",
):
    """
    Multi-head-style self-attention over the sequence.
    inputs: (T, B, D) time-major
    returns: (T, B, num_units)
    """
    with tf.variable_scope(scope):
        T = tf.shape(inputs)[0]
        B = tf.shape(inputs)[1]
        D = inputs.shape[-1].value or tf.shape(inputs)[2]

        # Project to Q, K, V
        Wq = tf.get_variable("Wq", [D, num_units], initializer=tf.glorot_uniform_initializer())
        Wk = tf.get_variable("Wk", [D, num_units], initializer=tf.glorot_uniform_initializer())
        Wv = tf.get_variable("Wv", [D, num_units], initializer=tf.glorot_uniform_initializer())

        # (T,B,D) @ (D,U) -> (T,B,U)
        q = tf.tensordot(inputs, Wq, axes=[[2], [0]])
        k = tf.tensordot(inputs, Wk, axes=[[2], [0]])
        v = tf.tensordot(inputs, Wv, axes=[[2], [0]])

        # Scaled dot-product: (T,B,U) * (T,B,U)^T over T -> (T,B,T) then (T,B,T) @ (T,B,U) -> (T,B,U)
        # Reshape to (B,T,U) for batch matmul: (B,T,U) @ (B,U,T) -> (B,T,T)
        qb = tf.transpose(q, [1, 0, 2])
        kb = tf.transpose(k, [1, 0, 2])
        vb = tf.transpose(v, [1, 0, 2])
        logits = tf.matmul(qb, kb, transpose_b=True) / (tf.cast(num_units, tf.float32) ** 0.5)
        attn = tf.nn.softmax(logits, axis=-1)
        out_b = tf.matmul(attn, vb)
        out = tf.transpose(out_b, [1, 0, 2])

        out = tf.layers.dropout(out, rate=dropout, training=is_training)
        return out


def hrnn_layer(
    inputs,
    num_units: int,
    is_training,
    dropout: float = 0.2,
    scope: str = "hrnn",
):
    """
    Handwriting Reconstruction RNN: 1-layer unidirectional GRU to refine
    stroke/alignment/spacing at the feature level.
    inputs: (T, B, D) time-major
    returns: (T, B, num_units)
    """
    with tf.variable_scope(scope):
        cell = tf.nn.rnn_cell.GRUCell(num_units)
        keep_prob = tf.cond(
            is_training,
            lambda: tf.constant(1.0 - dropout, dtype=tf.float32),
            lambda: tf.constant(1.0, dtype=tf.float32),
        )
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        outputs, _ = tf.nn.dynamic_rnn(
            cell,
            tf.transpose(inputs, [1, 0, 2]),
            dtype=tf.float32,
            time_major=False,
        )
        return tf.transpose(outputs, [1, 0, 2])


def feature_enhancement_block(
    bilstm_output,
    seq_length,
    num_units: int = 256,
    is_training=None,
    attention_units: int = 128,
    attention_dropout: float = 0.1,
    hrnn_dropout: float = 0.2,
    scope: str = "feature_enhancement",
):
    """
    Full Feature Enhancement: Self-Attention + residual + HRNN.

    Args:
        bilstm_output: (T, B, D) time-major, e.g. BiLSTM concat output
        seq_length: (B,) actual sequence lengths
        num_units: HRNN hidden size and output projection
        is_training: bool or scalar tf.placeholder
        attention_units: size for Q,K,V in attention
        attention_dropout: dropout after attention
        hrnn_dropout: dropout in HRNN

    Returns:
        enhanced: (T, B, num_units)
    """
    with tf.variable_scope(scope):
        T = tf.shape(bilstm_output)[0]
        B = tf.shape(bilstm_output)[1]
        D = bilstm_output.shape[-1].value or tf.shape(bilstm_output)[2]

        # Self-attention over the sequence
        attn_out = self_attention_layer(
            bilstm_output,
            num_units=attention_units,
            is_training=is_training,
            dropout=attention_dropout,
            scope="attention",
        )

        # Project attention to match D if needed, then residual
        if attention_units != D:
            Wproj = tf.get_variable(
                "attn_proj", [attention_units, D],
                initializer=tf.glorot_uniform_initializer(),
            )
            attn_out = tf.tensordot(attn_out, Wproj, axes=[[2], [0]])
        combined = bilstm_output + attn_out

        # Project to num_units for HRNN if D != num_units
        if D != num_units:
            W_in = tf.get_variable(
                "hrnn_in", [D, num_units],
                initializer=tf.glorot_uniform_initializer(),
            )
            combined = tf.tensordot(combined, W_in, axes=[[2], [0]])

        # HRNN refinement
        enhanced = hrnn_layer(
            combined,
            num_units=num_units,
            is_training=is_training,
            dropout=hrnn_dropout,
            scope="hrnn",
        )
        return enhanced
