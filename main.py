from train import train
from pre_train import pre_train
import optuna
import pickle

if __name__ == "__main__":
    def objective(trial):
        hyperparameters = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log = True),
            "dim_hidden": trial.suggest_int("dim_hidden", 32, 1024, step = 32),
            "num_blocks": trial.suggest_int("num_blocks", 1, 12),
            "mask_ratio": trial.suggest_float("mask_ratio", 0.1, 0.8),
            "epochs_cls": trial.suggest_int("epochs_cls", 10, 100),
            "lr_cls": trial.suggest_float("lr_cls", 1e-5, 1e-3, log = True),
            "model_type": trial.suggest_categorical("model_type_cls", ["transformer", "mlp"]),
        }
        if hyperparameters["model_type"] == "transformer":
            hyperparameters["compression"] = trial.suggest_categorical("compression",[16,32,64])
        print(hyperparameters)
        model = pre_train(hyperparameters)
        scores = train(hyperparameters,model )
        return scores
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    with open("sampler.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)
    print(study.best_params)
    print(study.best_value)
