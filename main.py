from train import train
from pre_train import pre_train
import optuna
# lr = 1e-4
# lr_cls = 1e-4
# epochs = 10
# epochs_cls = 100
# dim_hidden = 256
# num_blocks = 4
# compression = 32
# model_type  = "transformer"
# mask_ratio = 0.15

if __name__ == "__main__":
    def objective(trial):
        hyperparameters = {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
            "dim_hidden": trial.suggest_int("dim_hidden", 32, 1024, log=True),
            "num_blocks": trial.suggest_int("num_blocks", 1, 12),
            "compression": trial.suggest_categorical("compression",[8,16,32,64]),
            "mask_ratio": trial.suggest_float("mask_ratio", 0.1, 0.8),
            "epochs_cls": trial.suggest_int("epochs_cls", 10, 100),
            "lr_cls": trial.suggest_loguniform("lr_cls", 1e-5, 1e-3),
            "model_type": trial.suggest_categorical("model_type_cls", ["transformer", "mlp"]),
        }
        print(hyperparameters)
        model = pre_train(hyperparameters)
        scores = train(hyperparameters,model )
        return scores
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
