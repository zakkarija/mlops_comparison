# Failure Prediction in Manufacture

## Folders

* `binary-classification-model`: This is **DEPRECATED**, so don't use this code. Use the code of the *multiclass-classification-model* folder instead.
* `data_subset`: Contains datasets (separated by date) of a backward-forward Y axis movement of a grinding machine.
* `multiclass-classification-model`: Deep Learning models for mutilclass classification of timeseries.

## Dataset description

High frequency data is generated periodically by the machine’s control. The data is collected at a rapid pace when the machine performs a series of predefined, non-destructive movements, without machining.

In the case of the hysteresis tests of the axes, the axis performs linear backward and forward movements. The movement consist of moving the red part, called table, of the following figure. What moves this table is the ball screw.

<p align="center">
  <img src="https://i.imgur.com/Ts6eFsn.png" />
</p>

The same run is performed 3 times, so there are 3 forward (movement with negative intensity) and 3 backward (movement with positive intensity) movements in each test.

The following signal are captured in these tests:
- **Encoder position (*f1*)**: position in which the table should be, which is deduced from the angular position of the motor.
- **Ruler position (linear) (*f2*)**: position in which the table is actually located.
- **Intensity/current (*f3*)**: current supplied to the motor to make it move. This is the indicator we take into account to know if data is anomalous or not.
- **Commanded position (*f4*)**: the position that the motors are commanded from the control.

## Type of anomalies/data

An example of each category or type of anomalies is shown below.

In the following images the blue background sections are the ones where the axis is performing a forward movement and the orange background sections are the ones where the axis is performing a backward movement.

The axis moves from one side to the other one until it reaches a maximum, and then the movement towards the other side begins. This can be seen in the position signals in the plots (orange line) that are shown below. The peak of the position signal is when the axis changes the direction of the movement (from backward to forward or the other way around).

The intensity signal is the one plotted in blue. There is no a threshold indicating whether the data is anomalous or not, but it has been observed that the evolution of the intensity signal is different from anomalous data to non-anomalous data.

### 1. Not anomalous data

When the machine and its components do not have any faults this is the evolution of the intensity signal (blue). This signal takes values around 4 and -4 .

<p align="center">
  <img src="https://i.imgur.com/WUFeh48.png"/>
</p>

### 2. Mechanical components failure data (bearing and screw):

This is data example when a mechanical component of the machine is faulty. It can be observed that the intensity signal has more noise in the backward movements (orange background sections) than in forward movements (blue background sections). This is due to the fact that a component on one side of the ball screw is broken, which means that the intensity consumption is not the same in both directions.

<p align="center">
  <img src="https://i.imgur.com/I9vGl1B.png"/>
</p>

The signals analyzed to detect mechanical anomalies are:
* `f3` (intensity/current)
* It would be interesting to analyze the **looseness**, that is, the difference between `f2` (ruler position) and `f1` (motor position)
* It would be interesting to analyze the **tracking error**, that is, the difference between `f4` (commanded position) and `f2` (ruler position)

### 3. Electrical devices failure data (motors and drive):

This is a data example when there is an electrical failure. It can be observed that the intensity signal has a larger peak at the beginning of each section compared to non-anomalous data (these peaks take values around 10 and -10). This means that the motor needs more energy for starting the movement. In addition, the three peaks in the backward movements or the three peaks in the forward movements take different values, they are not constant or consistent, and this means that there is a failure in the acceleration.

<p align="center">
  <img src="https://i.imgur.com/0GczwV0.png"/>
</p>

The signals analyzed to detect electrical anomalies are:
* `f3` (intensity/current)
* It would be interesting to analyze the **tracking error**, that is, the difference between `f4` (commanded position) and `f2` (ruler position)

## Data subset

The folder `data_subset` contains a subset of the data. There are three sub-folders, one for each type of data/anomaly described in the previous section. The files are separated into folders by days in these sub-folders.

## Size of the dataset

Self-diagnostic cycles are performed every X time, so these data and files are generated with this frequency. The further we go on time, the more the data we will have. Therefore, it depends on how much data we need and until what date we want to take the data into account.

## Content of the files

The `.csv` file contains data gathered from the machine every 0.002 seconds (see `time` column below). This means we gather 500 values per second.

<p align="center">
  <img src="https://i.imgur.com/Kw8WG5O.png" />
</p>

About the data:

* The `time` column is not a *date*, it's a sampling value too (every 0.002 seconds) and it always starts from `0`.
* The length of the file can vary depending on duration the movement. This means that the machine is instructed not with specific time-based commands like "move for 30 seconds and record data". The operator configures the backward-forward movement and the machines takes the time it needs to execute them.
* When the movement is done. The machine generates one `.csv` (zipped) with a concrete filename (see `Filename` section).
* The trigger to make a `backward-forward` movement (= generated one file) on the machine is programed by the machine operator and is triggered multiple times a day.
* If 10 backward-forward movement are done, 10 files (like the ones you can find on the `data_subset` subfolders) are generated.

## Filename

The filename has the following form `1673136924134-2023_01_08_00_15_24_COD020030.zip` where:

* `1673136924134`: The time when data was captured in milliseconds.
* `2023_01_08_00_15_24`: The time when data was captured in `YYYY_MM_DD_HH_MM_SS` format.
* `COD020030`: Internal code to discribe the timeseries. `2` indicades the axis number (`2` = `Y axis`) and `30` indicates the cycle code (`30` = `backward-forward`)

## The reason for using Deep Learning models and not statistical features

The files labelled as anomalous data, particularly mechanical anomalous data, can be distinguished from not anomalous data just by extracting statistical features from the signal of the current intensity (`f3`). This is because the mechanical anomalous data was captured when the machine was broken, a mechanical component of the machine was completely destroyed. Hence, we do not have a degradation in the data and neither data on the deterioration process of the component. **The data we have correspond to a very good or very bad state of the machine, two completely different machine states**.

When the machine starts to deteriorate, it can be difficult to detect this scenario just by analysing the statistical features. **These statistics may not be representative for the deterioration process and this is the phase we are most interested in detecting in order to be able to repair the machine and avoid defective parts.** That is why we are interested in analysing the whole timeseries and use Deep Learning models for anomaly detection.

Besides, due to lubrication and temperature the statistics extracted from the signal of the current intensity may vary even though the machine has the same “state of health” and these features may noy be enough to detect anomalies.

## TODOs

- IDEKO: Change from binary classification (anomaly/no anomaly) to multiclass classification (mechanical anomaly/electrical anomaly/no anomaly).
- IDEKO: Analyze the looseness and the tracking error for the different types of data/anomalies we have.
- Andreas, Tomas: How and where to train RNN and LSTM models.