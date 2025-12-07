# nnclassifiers_experimental.py
# Implementation of Structured Self-Attentive Sentence Embedding (Lin et al., ICLR 2017)
#
# Lin, Z., Feng, M., Nogueira dos Santos, C., Yu, M., Xiang, B., Zhou, B., & Bengio, Y. (2017).
# A Structured Self-attentive Sentence Embedding. ICLR 2017. http://arxiv.org/abs/1703.03130

import json
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    TimeDistributed,
    Lambda,
    Flatten,
    LSTM,
    Dense,
    Bidirectional,
    Embedding,
)
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.losses import CategoricalCrossentropy

from embeddings import WordEmbeddings
from nnclassifiers import SimpleLSTMTokenizedDocumentClassifier, ConversionHelpers
from tcframework import TokenizedDocument, LabeledTokenizedDocument
from vocabulary import Vocabulary


# ----------------------------------------------------------------------
# Penalization term from Lin et al.
# ----------------------------------------------------------------------
class PRegularizer(Regularizer):
    """Implements ||AAᵀ - I||²_F penalization."""

    def __init__(self, coefficient: float = 1.0):
        self.coefficient = float(coefficient)

    def __call__(self, x):
        # x: (batch, timesteps, r)
        AAT = tf.matmul(x, x, transpose_a=True)  # (batch, r, r)

        r_dim = AAT.shape[-1]
        I = tf.eye(r_dim)

        diff = AAT - I
        fro_norm_sq = tf.square(tf.norm(diff, ord="fro", axis=[-2, -1]))

        return tf.reduce_sum(self.coefficient * fro_norm_sq)

    def get_config(self):
        return {"coefficient": float(self.coefficient)}


# ----------------------------------------------------------------------
# Main SSAE classifier
# ----------------------------------------------------------------------
class StructuredSelfAttentiveSentenceEmbedding(SimpleLSTMTokenizedDocumentClassifier):
    """
    Self-attentive sentence embedding model used in Task 3 (AH vs Delta).
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        embeddings: WordEmbeddings,
        output_dir_data_analysis: str = None,
    ):
        super().__init__(vocabulary, embeddings)
        self._output_dir_data_analysis = output_dir_data_analysis

        if output_dir_data_analysis and not os.path.exists(output_dir_data_analysis):
            os.makedirs(output_dir_data_analysis)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def get_model(self, numpy_matrix_embeddings: np.ndarray, **kwargs) -> Model:
        dropout = float(kwargs.get("dropout", 0.9))
        lstm_layer_size = int(kwargs.get("lstm_layer_size", 64))

        param_da = 300  # hidden attention dimension
        param_r = 50    # number of attention heads
        penalization_coefficient = 0.0  # Matches original code

        input_sequence = Input(shape=(None,), dtype="int32", name="input_sequence")

        embedded = Embedding(
            input_dim=numpy_matrix_embeddings.shape[0],
            output_dim=numpy_matrix_embeddings.shape[1],
            weights=[numpy_matrix_embeddings],
            mask_zero=True,
            trainable=False,
            name="embeddings",
        )(input_sequence)

        # BiLSTM → sequence output
        lstm_output = Bidirectional(
            LSTM(lstm_layer_size, return_sequences=True),
            name="BiLSTM",
        )(embedded)

        # Ws1 * H -> tanh
        H_tanh = Dense(
            units=param_da,
            activation="tanh",
            use_bias=False,
            name="tanh_Ws1_HT",
        )(lstm_output)

        # Ws2 * (tanh output)
        A_linear = Dense(
            units=param_r,
            activation="linear",
            use_bias=False,
            name="A_matrix",
        )(H_tanh)

        # Softmax along time dimension
        A = Lambda(
            lambda x: tf.nn.softmax(x, axis=1),
            name="A_softmax",
            activity_regularizer=PRegularizer(penalization_coefficient),
        )(A_linear)

        # M = AᵀH  → shape (batch, r, 2*lstm_layer_size)
        M = Lambda(
            lambda x: tf.matmul(x[0], x[1], transpose_a=True),
            name="M_matrix",
        )([A, lstm_output])

        flat = Flatten(name="flatten_M")(M)

        output = Dense(2, activation="softmax", name="Output_dense")(flat)

        model = Model(inputs=[input_sequence], outputs=output)
        model.summary()

        model.compile(
            optimizer="adam",
            loss=CategoricalCrossentropy(),
        )

        print("Structured Self-Attentive Sentence Embedding compiled.")
        return model

    # ------------------------------------------------------------------
    # Test + optional attention export + save final fold probabilities
    # ------------------------------------------------------------------
    def test(self, document_list: list, **kwargs) -> list:
        # 1) Get predicted labels (base class behavior)
        result = super().test(document_list, **kwargs)

        # 2) Only compute x_test once if we need it
        x_test = None
        if self._output_dir_data_analysis or kwargs.get("fold_no"):
            x_test = ConversionHelpers.convert_all_instances_to_x_vectors(
                document_list, self._max_length, self._vocabulary
            )

        # 3) Save attention visualizations per fold (if requested)
        if self._output_dir_data_analysis and kwargs.get("fold_no"):
            fold = int(kwargs["fold_no"])
            out_file = os.path.join(self._output_dir_data_analysis, f"fold{fold}.json")

            att_layer = Model(
                inputs=self._model.input,
                outputs=self._model.get_layer("A_softmax").output,
            )
            A_output = att_layer.predict({"input_sequence": x_test}, verbose=0)

            storage = []

            for i, doc in enumerate(document_list):
                A_matrix = A_output[i]              # (timesteps, r)
                weights = A_matrix.sum(axis=1)      # sum over attention heads
                weights = weights[: len(doc.tokens)]  # trim padding

                tokens, norm_weights = self.debug_weights_with_words(doc, weights)

                storage.append(
                    {
                        "id": doc.id,
                        "gold": doc.label,
                        "predicted": result[i],
                        "words": tokens,
                        "weights": norm_weights,
                    }
                )

            with open(out_file, "w") as f:
                json.dump(storage, f)
                print(f"Saved attention visualization to: {out_file}")

        # 4) On **last fold only** (fold 10) save raw probabilities for later BERT fusion
        if kwargs.get("fold_no"):
            fold = int(kwargs["fold_no"])
            if fold == 10:
                probs = self._model.predict({"input_sequence": x_test}, verbose=0)
                np.save("saved_ssae_probs.npy", probs)
                print("[SSAE] Final fold probabilities saved → saved_ssae_probs.npy")

        return result

    # ------------------------------------------------------------------
    # Extra API: get P(AH) per document (for later BERT usage)
    # ------------------------------------------------------------------
    def predict_proba(self, document_list: list) -> np.ndarray:
        """
        Return P(AH) for each document in `document_list`.

        This is useful if you want to call the trained SSAE model separately
        and use its AH-score as an extra feature for your RoBERTa classifier.
        """
        if self._model is None:
            raise ValueError("Model is not trained yet. Call .train(...) first.")

        x_test = ConversionHelpers.convert_all_instances_to_x_vectors(
            document_list, self._max_length, self._vocabulary
        )

        # Probability distribution: shape (N, 2)
        probs = self._model.predict({"input_sequence": x_test}, verbose=0)

        # Determine which index corresponds to label "ah" (case-insensitive)
        ah_index = None
        for idx, label in self._int_to_label_map.items():
            if label.lower() == "ah":
                ah_index = idx
                break

        if ah_index is None:
            raise ValueError(
                f"'ah' label not found in int_to_label_map: {self._int_to_label_map}"
            )

        # Extract the AH probability column → shape (N,)
        ah_probs = probs[:, ah_index]
        return ah_probs

    # ------------------------------------------------------------------
    # Debug helper: match tokens with normalized weights
    # ------------------------------------------------------------------
    def debug_weights_with_words(
        self, doc: TokenizedDocument, weights: np.ndarray
    ):
        if np.max(weights) > np.min(weights):
            norm = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        else:
            norm = np.zeros_like(weights)

        tokens = doc.tokens
        weights = [float(w) for w in norm]

        return tokens, weights
