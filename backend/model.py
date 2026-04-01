import torch
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.common import ListDataset
from lag_llama.gluon.estimator import LagLlamaEstimator

DEVICE = "cpu"
CKPT_PATH = "lag-llama.ckpt"

predictor_cache = {}

def load_predictor(prediction_length, context_length):
    key = f"{prediction_length}_{context_length}"

    if key in predictor_cache:
        return predictor_cache[key]

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path=CKPT_PATH,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=args["input_size"],
        n_layer=args["n_layer"],
        n_embd_per_head=args["n_embd_per_head"],
        n_head=args["n_head"],
        scaling=args["scaling"],
        time_feat=args["time_feat"],
        batch_size=1,
        num_parallel_samples=20,
    )

    predictor = estimator.create_predictor(
        estimator.create_transformation(),
        estimator.create_lightning_module()
    )

    predictor_cache[key] = predictor
    return predictor


def run_forecast(df, prediction_length, context_lengths):
    df["date"] = df["date"].astype("datetime64[ns]")
    df = df.sort_values("date")

    dataset = ListDataset(
        [{"target": df["value"].values, "start": df["date"].min()}],
        freq="D"
    )

    results = {}

    for ctx in context_lengths:
        predictor = load_predictor(prediction_length, ctx)

        forecast_it, _ = make_evaluation_predictions(
            dataset=dataset,
            predictor=predictor,
            num_samples=20
        )

        forecast = list(forecast_it)[0]
        results[str(ctx)] = forecast.mean_ts.values.tolist()

    return results