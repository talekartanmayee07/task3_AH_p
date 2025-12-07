# reader.py
import json
import random
import glob

import preprocessing
from tcframework import LabeledTokenizedDocumentReader, LabeledTokenizedDocument


class JSONPerLineDocumentReader(LabeledTokenizedDocumentReader):
    """
    Reads JSONL files where each line is one comment instance.
    Mostly used for Task 1 and Task 2 in the original UKP code.
    """

    def read_instances(self, input_path: str, instances_limit: int = -1) -> list:
        result = []

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                result.append(self.line_to_instance(line))

        random.shuffle(result)

        if instances_limit > 0:
            return result[:instances_limit]
        return result

    def line_to_instance(self, line: str) -> LabeledTokenizedDocument:
        m = json.loads(line)

        result = LabeledTokenizedDocument()
        result.id = m.get("name", "")
        result.label = "AH" if m.get("violated_rule", None) == 2 else "None"

        text_without_quotes = JSONPerLineDocumentReader.replace_quoted_text_with_special_token(
            m.get("body", "")
        )

        # Tokenize using original UKP code
        result.tokens = preprocessing.tokenize(text_without_quotes)

        return result

    @staticmethod
    def replace_quoted_text_with_special_token(text: str) -> str:
        paragraphs = text.splitlines(keepends=False)
        replaced = []
        for p in paragraphs:
            if p.startswith(">"):
                replaced.append("__quoted_text__")
            else:
                replaced.append(p)
        return "\n".join(replaced)


class DeltaVersusAHMinus1Reader(JSONPerLineDocumentReader):
    """
    Reader for their second experiment (not needed for Task 3),
    but kept intact for compatibility.
    """

    def line_to_instance(self, line: str) -> LabeledTokenizedDocument:
        m = json.loads(line)

        result = LabeledTokenizedDocument()
        result.id = m.get("name", "")
        result.label = "Delta" if m.get("delta", False) else "AH-1"
        result.tokens = preprocessing.tokenize(m.get("body", ""))

        return result


class AHVersusDeltaThreadReader(LabeledTokenizedDocumentReader):
    """
    THE IMPORTANT READER FOR TASK 3
    --------------------------------
    Reads thread-level JSON files:
        - If filename contains "_ah_", label = "ah"
          and we DROP the final comment (the actual ad hominem)
        - If filename contains "_delta_", label = "delta"
        - We concatenate the last 3 comments (default UKP behavior)
        - Add marker tokens ___<name>___start__
        - Replace quoted text (">") with __quoted_text__
        - Tokenize the body with preprocessing.tokenize()

    Output is a LabeledTokenizedDocument whose `tokens` field is:
        [
            "___c1___start__", tokens_of_comment_1,
            "___c2___start__", tokens_of_comment_2,
            "___c3___start__", tokens_of_comment_3,
            ...
        ]
    """

    def read_instances(self, input_path: str, instances_limit: int = -1) -> list:
        json_files = glob.glob(input_path + "/*.json")
        result = []

        for json_file in json_files:
            instance = self.file_to_instance(json_file)
            if instance is not None:
                result.append(instance)

        random.shuffle(result)

        if instances_limit > 0:
            return result[:instances_limit]
        return result

    def file_to_instance(self, file_name: str) -> LabeledTokenizedDocument:
        relative_name = file_name.split("/")[-1]
        parts = relative_name.split("_", 2)

        if len(parts) < 3:
            raise ValueError(f"Unexpected filename format: {relative_name}")

        label = parts[1]      # "ah" or "delta"
        file_id = parts[2]    # rest of filename

        if label not in ["ah", "delta"]:
            raise ValueError(f"Invalid label in filename: {relative_name}")

        result = LabeledTokenizedDocument()
        result.label = label
        result.id = relative_name
        result.tokens = []    # initialize tokens list properly

        # Read all lines (each line = one comment JSON)
        with open(file_name, "r", encoding="utf-8") as f:
            lines = [line for line in f]

        # REMOVE last comment if AH file (their original logic!)
        if label == "ah" and len(lines) > 0:
            lines = lines[:-1]

        # For each of the remaining comment lines:
        # Add ___<name>___start__ and tokenize the body
        for line in lines:
            m = json.loads(line)

            marker = f"___{m.get('name', '')}___start__"
            result.tokens.append(marker)

            body = m.get("body", "")
            clean_body = JSONPerLineDocumentReader.replace_quoted_text_with_special_token(body)
            result.tokens.extend(preprocessing.tokenize(clean_body))

        return result


if __name__ == "__main__":
    # Quick test: list instances
    reader = AHVersusDeltaThreadReader()
    instances = reader.read_instances("data/sampled-threads-ah-delta-context3")
    print(f"Loaded {len(instances)} instances.")
