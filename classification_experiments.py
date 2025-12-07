from classifiers import AbstractTokenizedDocumentClassifier
from embeddings import WordEmbeddings
from nnclassifiers import StackedLSTMTokenizedDocumentClassifier
from nnclassifiers_experimental import StructuredSelfAttentiveSentenceEmbedding
from readers import JSONPerLineDocumentReader, AHVersusDeltaThreadReader
from tcframework import (
    LabeledTokenizedDocumentReader,
    AbstractEvaluator,
    Fold,
    TokenizedDocument,
    ClassificationEvaluator,
)
from vocabulary import Vocabulary

import json
import numpy as np


class ClassificationExperiment:
    def __init__(
        self,
        labeled_document_reader: LabeledTokenizedDocumentReader,
        classifier: AbstractTokenizedDocumentClassifier,
        evaluator: AbstractEvaluator,
    ):
        self.reader = labeled_document_reader
        self.classifier = classifier
        self.evaluator = evaluator

    def run(self) -> None:
        folds = self.reader.get_folds()

        for i, fold in enumerate(folds, start=1):
            print(f"Running fold {i}/{len(folds)}")

            # Train on fold i
            self.classifier.train(fold.train)

            # Predict fold i test set
            predicted_labels = self.classifier.test(fold.test, fold_no=i)

            # Evaluate fold i
            self.evaluate_fold(fold.test, predicted_labels)

            print(f"Evaluating after {i} folds")
            self.evaluator.evaluate()

        # Final evaluation on all folds
        print(f"Final evaluation; reader.input_path_train was {self.reader.input_path_train}")
        self.evaluator.evaluate()

        # ⬇️ NEW PART: SAVE PREDICTIONS FOR ALL DOCUMENTS (train + test, 2582 total)
        self.save_predictions_for_all_documents()

    # ----------------------------------------------------------
    # Save predictions for *all* documents (train + test)
    # ----------------------------------------------------------
    def save_predictions_for_all_documents(self):

        all_docs = self.reader.train  # All 2582 doc objects
        print(f"[INFO] Generating probabilities for ALL {len(all_docs)} documents...")

        # Use the SSAE probability API
        probs = self.classifier.predict_proba(all_docs)

        out = []
        for doc, p in zip(all_docs, probs):
            out.append({
                "id": doc.id,
                "prob_ah": float(p),
                "prob_delta": float(1 - p),
            })

        with open("all_ssae_predictions.json", "w") as f:
            json.dump(out, f, indent=2)

        print("[SSAE] Saved predictions for ALL documents → all_ssae_predictions.json")

    def evaluate_fold(self, labeled_document_instances: list, predicted_labels: list):
        assert len(labeled_document_instances) == len(predicted_labels)

        gold = [doc.label for doc in labeled_document_instances]
        ids = [doc.id for doc in labeled_document_instances]

        self.evaluator.add_single_fold_results(gold, predicted_labels, ids)


# ============================================================
# Cross-validation setups
# ============================================================

def cross_validation_thread_ah_delta_context3():
    import random
    import tensorflow as tf

    random.seed(1234567)
    tf.random.set_seed(1234567)

    vocabulary = Vocabulary.deserialize("en-top100k.vocabulary.pkl.gz")
    embeddings = WordEmbeddings.deserialize("en-top100k.embeddings.pkl.gz")

    reader = AHVersusDeltaThreadReader("data/sampled-threads-ah-delta-context3", True)

    experiment = ClassificationExperiment(
        reader,
        StructuredSelfAttentiveSentenceEmbedding(
            vocabulary,
            embeddings,
            output_dir_data_analysis="/tmp/visualization-context3",
        ),
        ClassificationEvaluator(),
    )
    experiment.run()


if __name__ == "__main__":
    cross_validation_thread_ah_delta_context3()
