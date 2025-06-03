# DEPRECATED. Use multiclass classification [code](https://colab-repo.intracom-telecom.com/colab-projects/extremexp/uc-data/uc5-ideko/failure-prediction-in-manufacture/-/tree/main/multiclass-classification-model?ref_type=heads]).

# Failure Prediction in Manufacture - binary classification models

Deep learning models for binary classification of timeseries.
* Neural Network
* Convolutional Neural Network (CNN)
* Recurrent Neural Network (RNN)
* Long Shot Term Memory (LSTM)

## Activating / deactivating models

Models can be activated/deactivated by modifying `enabled` flag on the `src/config/config_models.yaml` file.

```yaml
NeuralNetwork:
    enabled: True
```

You can also tune model parameters (to define models with different architecture) and training parameters on that file.

## Run the code

1. Clone the repository.
2. Edit config file `src/config/config.yaml` if necessary.
3. Build the Docker image:
```bash
cd binary-classification-model
docker build -t binary-classification-model .
```

4. Run the Docker container by mapping:
* The shared volume where the data is (data from the [data_subset folder](https://colab-repo.intracom-telecom.com/colab-projects/extremexp/uc-data/uc5-ideko/failure-prediction-in-manufacture/-/tree/main/data_subset?ref_type=heads)). Maintain the `YYYMMDD` subfolders.
* The shared volume where the result will be placed

```bash
docker run -v ./data:/app/data -v ./output:/app/output --name models binary-classification-model
```