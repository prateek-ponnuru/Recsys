from collections import defaultdict
import pandas as pd

# def get_predictions(model, user, movies, k):
#     movies['user'] = user
#     preds = movies.apply(lambda x: model.predict(x[0], x[-1]), 1, result_type='expand')
#     idx = preds[3].argsort()[:k]
#     ids = preds.iloc[idx, 0]
#     mvs = movies.movieId.isin(ids)
#     return movies.loc[mvs, ['title', 'genres']]

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, true_r, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def get_item_details(predictions, items, idcol, collist):

    ids = [int(x[0]) for x in predictions]
    tru_r = [x[1] for x in predictions]
    est_r = [x[2] for x in predictions]
    df = pd.DataFrame({idcol: ids, 'tru_r': tru_r, 'est_r': est_r})
    # mvs = movies.movieId.isin(ids)
    items = items.loc[items[idcol].isin(ids), collist]

    return items.merge(df, on=idcol)