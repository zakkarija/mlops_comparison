# Failure Prediction in Manufacture - multiclass classification models

Deep Learning models for multiclass classification of timeseries based in Python 3.10.14.

* Neural Network
* Convolutional Neural Network (CNN)
* Recurrent Neural Network (RNN)
* Long Shot Term Memory (LSTM)

## Activating / deactivating models

Models can be activated/deactivated by modifying `enabled` flag on the `src/config/models.yaml` file.

```yaml
NeuralNetwork:
    enabled: True
```

You can also tune model parameters (to define models with different architecture) and training parameters on that file.

## Run the code

1. Clone the repository.
2. Edit model parameters config file `src/config/models.yaml` if necessary.
3. Build the Docker image:
```bash
cd multiclass-classification-model
docker build -t multiclass-classification-model .
```

4. Run the Docker container by mapping:
* The shared volume where the data is (data from the [data_subset folder](https://colab-repo.intracom-telecom.com/colab-projects/extremexp/uc-data/uc5-ideko/failure-prediction-in-manufacture/-/tree/main/data_subset?ref_type=heads)). Maintain the `electrical_anomalies`, `mechanical_anomalies` and `not_anomalous` subfolders.
* The shared volume where the result will be placed

```bash
docker run -v ./data:/app/data -v ./output:/app/output --name models multiclass-classification-model
```