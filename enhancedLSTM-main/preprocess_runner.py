"""
Run SNLI preprocessing for the ESIM training scripts.
"""

import pickle
from pathlib import Path

from data import Preprocessor


def dump_pickle(data, output_path):
    with open(output_path, "wb") as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    base_dir = Path(__file__).resolve().parent
    snli_dir = base_dir.parent / "config" / "snli_1.0"

    train_path = snli_dir / "snli_1.0_train.txt"
    valid_path = snli_dir / "snli_1.0_dev.txt"
    test_path = snli_dir / "snli_1.0_test.txt"
    glove_path = base_dir / "glove.840B.300d.txt"

    required_files = [train_path, valid_path, test_path, glove_path]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required preprocessing input files:\n"
            + "\n".join(missing_files)
        )

    preprocessor = Preprocessor(
        lowercase=True,
        ignore_punctuation=True,
        labeldict={
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        },
    )

    print("Reading raw SNLI data...")
    train_data = preprocessor.read_data(train_path)
    valid_data = preprocessor.read_data(valid_path)
    test_data = preprocessor.read_data(test_path)

    print("Building word dictionary from training data...")
    preprocessor.build_worddict(train_data)

    print("Transforming datasets to indices...")
    transformed_train = preprocessor.transform_to_indices(train_data)
    transformed_valid = preprocessor.transform_to_indices(valid_data)
    transformed_test = preprocessor.transform_to_indices(test_data)

    print("Building embedding matrix...")
    embeddings = preprocessor.build_embedding_matrix(glove_path)

    print("Writing pickle files...")
    dump_pickle(transformed_train, base_dir / "train_data.pkl")
    dump_pickle(transformed_valid, base_dir / "valid_data.pkl")
    dump_pickle(transformed_test, base_dir / "test_data.pkl")
    dump_pickle(embeddings, base_dir / "embeddings.pkl")

    print("Done.")


if __name__ == "__main__":
    main()
