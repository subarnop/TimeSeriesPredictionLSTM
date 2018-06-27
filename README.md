## Time Series Prediction using LSTM Recurrent Neural Network

A time series is a series of data points indexed in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting/prediction is the use of a model to predict future values based on previously observed values. <br>
We work on a dataset consisting of <b> the daily wage of England </b> (1260-1994) in pounds. This dataset a very rough dataset and the biggest challenge is that when we break it into 3:1 Train and Test, the charecterstics changes highly in Test part. This stiff growth and changes in wage is mainly due to the Industrial Revolution during the 1820s.<br>

<p align="center"> <img width=420 src="https://github.com/Subarno/TimeSeriesPredictionLSTM/blob/master/img/data_plot.png"> </p>

Generally time series prediction is done using well established statistical models such as SMA, ARIMA, ARIMAX but we will approach it using Recurrent Neural Nets and show a convincing result. Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in their work. They work tremendously well on a large variety of problems, and are now widely used. 
<p align="center"> 
<img  src="https://github.com/Subarno/TimeSeriesPredictionLSTM/blob/master/img/model_lstm.png"> 
</p>
By traing the network for 100 epochs with 1 look back window we got an Root Mean square (RMSE) error of 1.99 which is quite an acceptable result. We then finetune network updating the look back window size and check the result training for 100 epochs. 

| __look back__ | __Train Score (RMSE)__ | __Test Score (RMSE)__ |
|---------------|------------------------|-----------------------|
| 01 | 0.57 | 1.99 |
| 03 | 0.55 | 1.70 |
| 05 | 0.58 | **1.58** |
| 10 | 0.54 | *17.01* |

Thus we can see that the **previous 5 years** have significant influence on the value of the wages in the dataset.
Here we display the resulting graphgs due to fine tuning the look back window:

<img  align = "left" width="320" src="https://github.com/Subarno/TimeSeriesPredictionLSTM/blob/master/img/result_plot_lstm_1.png"> 
<img  width="320" src="https://github.com/Subarno/TimeSeriesPredictionLSTM/blob/master/img/result_plot_lstm_3.png">
<img  width="320" src="https://github.com/Subarno/TimeSeriesPredictionLSTM/blob/master/img/result_plot_lstm_5.png">
<img  width="320" src="https://github.com/Subarno/TimeSeriesPredictionLSTM/blob/master/img/result_plot_lstm_10.png">

### Usage:
1. Download/Clone the repository and get the dependenies.
2. Set the desired look_back
```
$python3 lstm.py
```

<br>
 Happy To Code ... Happy To Live :octocat:
