from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise.trainset import Trainset
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic
from surprise import SVD, NMF
from surprise import accuracy


def get_trained_model(model_class: AlgoBase, train_set: Trainset, model_kwargs: dict = {}) -> AlgoBase:
    model = model_class(**model_kwargs)
    model.fit(train_set)
    return model

def evaluate_model(model: AlgoBase, test_set: [(int, int, float)]) -> dict:
    predictions = model.test(test_set)
    metrics_dict = {}
    metrics_dict['RMSE'] = accuracy.rmse(predictions, verbose=False)
    metrics_dict['MAE'] = accuracy.rmse(predictions, verbose=False)
    return metrics_dict

def train_and_evalute_model(model_class: AlgoBase, data: Trainset,
                                     test_size: float = 0.2,
                                     model_kwargs: dict = {}) -> (AlgoBase, dict):
    train_set, test_set = train_test_split(data, test_size, random_state=42)
    model = get_trained_model(model_class, train_set, model_kwargs)
    metrics_dict = evaluate_model(model, test_set)
    return model, metrics_dict