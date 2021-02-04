from surprise import Dataset
from surprise.dataset import DatasetAutoFolds
from pathlib import Path
from surprise import Reader


def load_ratings_from_surprise(name : str) -> DatasetAutoFolds:
    ratings = Dataset.load_builtin(name)
    return ratings

def load_ratings_from_file(name : str) -> DatasetAutoFolds:
    data_dir = Path(Path.cwd().parents[0], 'data', 'movielens', 'ml-latest-small')
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    ratings = Dataset.load_from_file(Path(data_dir, name), reader)
    return ratings


def get_data(name : str, from_surprise : bool = True) -> DatasetAutoFolds:
    data = load_ratings_from_surprise(name) if from_surprise else load_ratings_from_file(name)
    return data

