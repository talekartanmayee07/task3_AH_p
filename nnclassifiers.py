import sys
import numpy as np
import os
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Embedding,
    LSTM,
    Concatenate,
    Dropout,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import pad_sequences  # <- use this instead of preprocessing.sequence

from classifiers import AbstractTokenizedDocumentRegression, AbstractTokenizedDocumentClassifier
from embeddings import WordEmbeddings
from tcframework import LabeledTokenizedDocument, TokenizedDocument
from vocabulary import Vocabulary


# ============================================================
# Simple LSTM Classifier (Base class used by SSAE too)
# ============================================================

class SimpleLSTMTokenizedDocumentClassifier(AbstractTokenizedDocumentClassifier):
    def __init__(self, vocabulary: Vocabulary, embeddings: WordEmbeddings):
        super().__init__()
        self._model = None
        self._vocabulary = vocabulary
        self._embeddings = embeddings
    def save_model(self, path="saved_ssae_model"):
        """Save trained model for later inference."""
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, "model.keras")
        self._model.save(out_path)
        print(f"[SSAE] Model saved to: {out_path}")

    @staticmethod
    def split_to_batches_by_document_lengths(labeled_document_list: list, batch_size: int = 32) -> list:
        if len(labeled_document_list) < batch_size:
            print(
                f"Requested batch size {batch_size} but got only {len(labeled_document_list)} documents",
                file=sys.stderr,
            )

        s = sorted(labeled_document_list, key=lambda d: len(d.tokens))
        indices = list(range(len(s)))
        result = [s[i:i + batch_size] for i in indices[::batch_size]]

        for i in range(len(result) - 1):
            assert len(result[i]) == batch_size

        return result

    def train(self, labeled_document_list: list, validation: bool = True) -> None:
        super().train(labeled_document_list, validation)
        assert self._label_to_int_map
        assert self._max_length

        print("Max length in training data:", self._max_length)

        nb_epoch = 5

        model = self.get_model(
            self._embeddings.to_numpy_matrix(),
            max_input_length=self._max_length,
        )

        x_train, y_train = ConversionHelpers.convert_all_instances_to_x_y_vectors_categorical(
            labeled_document_list,
            self._max_length,
            self._vocabulary,
            self._label_to_int_map,
        )

        model.fit(
            {"input_sequence": x_train},
            y_train,
            epochs=nb_epoch,
            verbose=1,
            validation_split=0.1 if validation else 0.0,
        )

        self._model = model

        # SAVE MODEL (only once per full training)
        if hasattr(self, "save_model"):
            self.save_model()

    def test(self, document_list: list, **kwargs) -> list:
        assert self._int_to_label_map

        x_test = ConversionHelpers.convert_all_instances_to_x_vectors(
            document_list, self._max_length, self._vocabulary
        )

        predictions = self._model.predict({"input_sequence": x_test}, verbose=0)

        result = ConversionHelpers.convert_predicted_prob_dist_to_label(
            predictions, self._int_to_label_map
        )
        return result

    def get_model(self, numpy_matrix_embeddings: np.ndarray, **kwargs) -> Model:
        dropout = kwargs.get("dropout", 0.0)
        lstm_layer_size = kwargs.get("lstm_layer_size", 64)

        input_sequence = Input(shape=(None,), dtype="int32", name="input_sequence")

        embedded = Embedding(
            numpy_matrix_embeddings.shape[0],
            numpy_matrix_embeddings.shape[1],
            weights=[numpy_matrix_embeddings],
            mask_zero=True,
            trainable=False,
        )(input_sequence)

        lstm_output = Bidirectional(LSTM(lstm_layer_size))(embedded)
        dropout_layer = Dropout(dropout)(lstm_output)

        fully_connected = Dense(int(lstm_layer_size // 2), activation="relu")(dropout_layer)
        output = Dense(2, activation="softmax")(fully_connected)

        model = Model(inputs=[input_sequence], outputs=output)
        model.compile(
            optimizer="adam",
            loss=CategoricalCrossentropy(),
        )
        return model


# ============================================================
# Stacked LSTM
# ============================================================

class StackedLSTMTokenizedDocumentClassifier(SimpleLSTMTokenizedDocumentClassifier):
    def get_model(self, numpy_matrix_embeddings: np.ndarray, **kwargs) -> Model:
        dropout = kwargs.get("dropout", 0.9)
        lstm_layer_size = kwargs.get("lstm_layer_size", 64)

        input_sequence = Input(shape=(None,), dtype="int32", name="input_sequence")
        embedded = Embedding(
            numpy_matrix_embeddings.shape[0],
            numpy_matrix_embeddings.shape[1],
            weights=[numpy_matrix_embeddings],
            mask_zero=True,
            trainable=False,
        )(input_sequence)

        lstm1 = Bidirectional(LSTM(lstm_layer_size, return_sequences=True))(embedded)
        lstm2 = Bidirectional(LSTM(lstm_layer_size))(lstm1)

        dropout_layer = Dropout(dropout)(lstm2)
        fully_connected = Dense(int(lstm_layer_size // 2), activation="relu")(dropout_layer)
        output = Dense(2, activation="softmax")(fully_connected)

        model = Model(inputs=[input_sequence], outputs=output)
        model.compile(
            optimizer="adam",
            loss=CategoricalCrossentropy(),
        )

        return model


# ============================================================
# CNN Classifier
# ============================================================

class CNNTokenizedDocumentClassifier(SimpleLSTMTokenizedDocumentClassifier):
    def get_model(self, numpy_matrix_embeddings: np.ndarray, **kwargs) -> Model:
        dropout = kwargs.get("dropout", 0.9)

        input_sequence = Input(shape=(None,), dtype="int32", name="input_sequence")

        embedded = Embedding(
            numpy_matrix_embeddings.shape[0],
            numpy_matrix_embeddings.shape[1],
            weights=[numpy_matrix_embeddings],
            mask_zero=False,
            trainable=False,
        )(input_sequence)

        nb_filters = 500
        kernel_sizes = [5, 7, 9, 11]

        conv_layers = [
            Conv1D(filters=nb_filters, kernel_size=k, padding="same")(embedded)
            for k in kernel_sizes
        ]

        concatenated = Concatenate()(conv_layers)
        pooled = GlobalMaxPooling1D()(concatenated)
        after_dropout = Dropout(dropout)(pooled)

        output = Dense(2, activation="softmax")(after_dropout)

        model = Model(inputs=[input_sequence], outputs=output)
        model.compile(
            optimizer="adam",
            loss=CategoricalCrossentropy(),
        )

        return model


# ============================================================
# Conversion Helpers
# ============================================================

class ConversionHelpers:
    @staticmethod
    def convert_single_instance_to_x_vector(doc: TokenizedDocument, vocabulary: Vocabulary) -> np.ndarray:
        return np.array(
            [
                vocabulary.word_to_index_mapping.get(
                    w, vocabulary.word_to_index_mapping[vocabulary.oov]
                )
                for w in doc.tokens
            ],
            dtype=np.int32,
        )

    @staticmethod
    def convert_single_instance_to_y_vector_categorical(
        doc: LabeledTokenizedDocument, label_to_int_map: dict
    ) -> np.ndarray:
        y = np.zeros(len(label_to_int_map), dtype=np.float32)
        y[label_to_int_map[doc.label]] = 1.0
        return y

    @staticmethod
    def convert_all_instances_to_x_vectors(
        instances: list, max_length: int, vocabulary: Vocabulary
    ) -> np.ndarray:
        x = [
            ConversionHelpers.convert_single_instance_to_x_vector(doc, vocabulary)
            for doc in instances
        ]
        # Keras 3: use tf.keras.utils.pad_sequences
        return pad_sequences(x, maxlen=max_length, padding="post", truncating="post")

    @staticmethod
    def convert_all_instances_to_x_y_vectors_categorical(
        instances: list,
        max_length: int,
        vocabulary: Vocabulary,
        label_to_int_map: dict,
    ):
        X = ConversionHelpers.convert_all_instances_to_x_vectors(
            instances, max_length, vocabulary
        )
        Y = np.array(
            [
                ConversionHelpers.convert_single_instance_to_y_vector_categorical(
                    doc, label_to_int_map
                )
                for doc in instances
            ],
            dtype=np.float32,
        )
        return X, Y

    @staticmethod
    def convert_predicted_prob_dist_to_label(
        y_predictions: np.ndarray, int_to_label_map: dict
    ) -> list:
        argmax_indices = np.argmax(y_predictions, axis=1)
        return [int_to_label_map[i] for i in argmax_indices]
