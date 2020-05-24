# Forecasting Stock Price using Deep Learning tools

The main goal of this project is to build a stock trading system, based on predicting the stock market prices and movements.

# Contributors

* [ABEDAbir](https://github.com/ABEDabir)
* [adnaneaabbar](https://github.com/adnaneaabbar)

# Snapshot

Since a real-time prediction which gonna result into a decision of buying or selling the stock, depend almost all the time on the few seconds or minutes of data that precede the time of decision, it was only logical to try and find a small dataset to work on to be closer to real trading process.

---

# Source of data : 

[Yahoo Finance](https://finance.yahoo.com/)

Small Data : [Goldman Sachs Inc.](https://github.com/adnaneaabbar/stock-trading-system/blob/master/data/goldman_sachs.csv)
Small Data : [General Electrics](https://github.com/adnaneaabbar/stock-trading-system/blob/master/data/ge.csv)

---

# Notebooks

Please find the notebooks with the details and steps for this project over here :

* [Intuitive LSTM Method on GS stocks (Small Data)](https://github.com/adnaneaabbar/stock-trading-system/blob/master/Intuitive_Method_On_Small_Data.ipynb)

Result :
![](https://github.com/adnaneaabbar/stock-trading-system/blob/master/pred/lstm_sd.png?raw=true)

We tested the same model for large data this time :

* [Intuitive LSTM Method on GE stocks (Large Data)](https://github.com/adnaneaabbar/stock-trading-system/blob/master/Testing_Intuitive_Method_On_Large_Data.ipynb)
Result :

![](https://github.com/adnaneaabbar/stock-trading-system/blob/master/pred/lstm_ld.png?raw=true)

---

### The prediction was nowhere near good with small data, so we looked for an optimized method :

* [Optimized Method on GS stocks (Small Data)](https://github.com/adnaneaabbar/stock-trading-system/blob/master/Optimized_Methods_On_Small_Data.ipynb)

Result : 
![](https://github.com/adnaneaabbar/stock-trading-system/blob/master/pred/opt_arima_sd.png?raw=true)

The results were nearly perfect, it didn't stop us from testing it on large data too :

* [Optimized Method on GE stocks (Large Data)](https://github.com/adnaneaabbar/stock-trading-system/blob/master/Testing_Optimized_Methods_On_Large_Data.ipynb)

Result :
![](https://github.com/adnaneaabbar/stock-trading-system/blob/master/pred/opt_arima_ld.png?raw=true)

---

# References

[deeplearning.ai : Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning?)

[deeplearning.ai : Tensorflow in Practice Specialization](https://www.coursera.org/specializations/tensorflow-in-practice)

[Hong Kong University of Science and Technology : Python and Statistics for Financial Analysis](https://www.coursera.org/learn/python-statistics-financial-analysis)



