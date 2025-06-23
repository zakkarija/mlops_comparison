import optuna
from optuna_integration.keras import KerasPruningCallback
import logging

logger = logging.getLogger(__name__)

class OptunaHelper:
    """Simple Optuna helper for model hyperparameter optimization"""
    
    @staticmethod
    def create_study(model_name, study_name="ideko_anomaly_detection"):
        """Create or load Optuna study"""
        return optuna.create_study(
            study_name=f"{study_name}_{model_name}",
            direction="maximize",
            storage="sqlite:///optuna_studies.db",
            load_if_exists=True
        )
    
    @staticmethod
    def suggest_hyperparams(trial, model_type, base_config):
        """Suggest hyperparameters based on model type"""
        params = {}
        
        if model_type == "NeuralNetwork":
            params['units'] = [trial.suggest_int(f"units_{i}", 32, 256) 
                              for i in range(len(base_config["model_parameters"]["units"]))]
            params['activation'] = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
            
        elif model_type == "CNN":
            params['filters'] = [trial.suggest_int(f"filters_{i}", 32, 256) 
                                for i in range(len(base_config["model_parameters"]["filters"]))]
            params['kernel_size'] = trial.suggest_int("kernel_size", 2, 5)
            params['pool_size'] = trial.suggest_int("pool_size", 2, 4)
            params['activation'] = trial.suggest_categorical("activation", ["relu", "tanh"])
            
        elif model_type in ["RNN", "LSTM"]:
            params['hidden_units'] = [trial.suggest_int(f"hidden_units_{i}", 32, 256) 
                                     for i in range(len(base_config["model_parameters"]["hidden_units"]))]
            params['activation'] = trial.suggest_categorical("activation", ["tanh", "relu"])
        
        # Common parameters for all models
        params['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        params['batch_size'] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        
        return params
    
    @staticmethod
    def get_keras_callback(trial):
        """Get Optuna Keras pruning callback"""
        return KerasPruningCallback(trial, "val_accuracy")
    
    @staticmethod
    def optimize_model(model_class, model_type, base_config, X_train, Y_train, X_test, Y_test, 
                      n_ts, n_feat, n_cls, n_trials=10):
        """
        Optimize model with Optuna
        Returns: (best_params, study)
        """
        logger.info(f"🔍 Starting Optuna optimization for {model_type}...")
        
        def objective(trial):
            # Get suggested hyperparameters
            params = OptunaHelper.suggest_hyperparams(trial, model_type, base_config)
            
            # Create model with suggested params
            if model_type == "NeuralNetwork":
                model = model_class(n_ts, n_feat, params['activation'], params['units'], n_cls)
            elif model_type == "CNN":
                model = model_class(n_ts, n_feat, params['activation'], params['filters'], 
                                  params['kernel_size'], params['pool_size'], n_cls)
            elif model_type in ["RNN", "LSTM"]:
                model = model_class(n_ts, n_feat, params['activation'], params['hidden_units'], n_cls)
            
            model.create_model()
            model.model_compilation(model.model)  # Use default compilation first
            
            # Re-compile with suggested learning rate
            from tensorflow.keras.optimizers import Adam
            model.model.compile(
                optimizer=Adam(learning_rate=params['learning_rate']),
                loss=model.model.loss,
                metrics=model.model.metrics
            )
            
            # Train with Optuna callback
            try:
                history = model.model.fit(
                    X_train, Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=min(20, base_config["training_parameters"]["epochs"]),
                    batch_size=params['batch_size'],
                    callbacks=[
                        OptunaHelper.get_keras_callback(trial),
                        model.early_stopping_callback(patience=3)
                    ],
                    verbose=0
                )
                return max(history.history.get('val_accuracy', [0]))
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Create and run study
        study = OptunaHelper.create_study(model_type)
        study.optimize(objective, n_trials=n_trials, timeout=3600)
        
        logger.info(f"✅ Best accuracy: {study.best_value:.4f}")
        logger.info(f"✅ Best params: {study.best_params}")
        
        return study.best_params, study