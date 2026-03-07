import os
import logging
import pandas as pd
from river.datasets import synth
from river.datasets import base
from river import stream
from river import datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_DIR = "data/stream_datasets"

class InsectsBalanced(base.FileDataset):
    def __init__(self):
        super().__init__(
            filename="INSECTS-abrupt_balanced_norm.csv",
            directory="data/other_datasets",
            n_samples=52_848,
            n_features=33,
            task=base.MULTI_CLF,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="Class",
        )

def generate_stream_to_dataframe(dataset, num_examples):
    """
    Generate examples from a synthetic dataset and return as a pandas DataFrame.
    
    Args:
        dataset: An initialized SyntheticDataset object from river.datasets.synth
        num_examples: The number of examples to generate
        
    Returns:
        pandas.DataFrame: DataFrame containing the generated examples with features and target
    """
    # Collect all examples
    examples = list(dataset.take(num_examples))
    
    if not examples:
        logger.warning("No examples generated from dataset")
        return pd.DataFrame()
        
    logger.debug(f"Generated {len(examples)} examples from dataset")
    
    # Determine number of features from first example
    num_features = len(examples[0][0])
    
    # Create column names
    feature_columns = [list(examples[0][0])[i] for i in range(num_features)]
    columns = feature_columns + ["target"]
    
    # Build data rows
    data = []
    for x, y in examples:
        row = [x[feature_columns[i]] for i in range(num_features)] + [y]
        data.append(row)
        
    # Create and return DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df


def process_stream(stream, stream_name, stream_size, drift_position):
    """Process a stream by generating data and saving to CSVs (pre and post drift)."""
    logger.info(
        f"Processing stream: {stream_name} (size={stream_size}, drift_pos={drift_position})"
    )

    df = generate_stream_to_dataframe(stream, stream_size)

    # Save pre-drift data
    pre_csv = f"{TARGET_DIR}/{stream_name}_pre.csv"
    df.iloc[:drift_position, :].to_csv(pre_csv, index=False)
    logger.info(f"Saved {pre_csv}")

    # Save post-drift data
    post_csv = f"{TARGET_DIR}/{stream_name}_post.csv"
    df.iloc[drift_position:, :].to_csv(post_csv, index=False)
    logger.info(f"Saved {post_csv}")


def main():
    logger.info("Starting synthetic stream generation")

    # Create directories if they don't exist
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    stream_size = 4000
    drift_position = 2000

    sea = synth.ConceptDriftStream(
        stream=synth.SEA(seed=23, variant=2),
        drift_stream=synth.SEA(seed=23, variant=3),
        seed=1,
        position=2000,
        width=50,
    )
    process_stream(sea, "sea", stream_size, drift_position)

    rbf = synth.ConceptDriftStream(
        stream=synth.RandomRBF(
            seed_model=23, seed_sample=23, n_classes=2, n_features=7, n_centroids=4
        ),
        drift_stream=synth.RandomRBF(
            seed_model=42, seed_sample=23, n_classes=2, n_features=7, n_centroids=4
        ),
        seed=1,
        position=2000,
        width=50,
    )
    process_stream(rbf, "rbf", stream_size, drift_position)

    hyp = synth.ConceptDriftStream(
        stream=synth.Hyperplane(
            seed=23,
            n_features=5,
            n_drift_features=2,
        ),
        drift_stream=synth.Hyperplane(
            seed=42,
            n_features=5,
            n_drift_features=2,
        ),
        seed=1,
        position=2000,
        width=50,
    )
    process_stream(hyp, "hyp", stream_size, drift_position)

    tree = synth.ConceptDriftStream(
        stream=synth.RandomTree(
            seed_tree=23,
            seed_sample=23,
            n_classes=4,
            n_num_features=9,
            n_cat_features=0,
            max_tree_depth=3,
            first_leaf_level=2,
            fraction_leaves_per_level=0.1,
        ),
        drift_stream=synth.RandomTree(
            seed_tree=42,
            seed_sample=42,
            n_classes=4,
            n_num_features=9,
            n_cat_features=0,
            n_categories_per_feature=2,
            max_tree_depth=6,
            first_leaf_level=3,
            fraction_leaves_per_level=0.2,
        ),
        seed=1,
        position=2000,
        width=50,
    )
    process_stream(tree, "tree", stream_size, drift_position)

    friedman = synth.FriedmanDrift(
        drift_type="gra", position=(2000, 4000), seed=23, transition_window=10
    )
    process_stream(friedman, "friedman", stream_size, drift_position)

    elec = datasets.Elec2()
    process_stream(elec, "elec", 20000, 10000)

    keystroke = datasets.Keystroke()
    process_stream(keystroke, "keystroke", 1600, 800)

    insects = InsectsBalanced()
    process_stream(insects, "insects", 52848, 26424)

    logger.info("All synthetic streams processed successfully")


if __name__ == "__main__":
    main()
