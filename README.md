# ID2223
## Project: Market Stock Prediction Using Sentiment Analysis
**Professor:**
Jim Dowling

**Students:**
- Yilin Chang
- Zineb Senane

## Task description

## Infrastructure Model
- Feature Pipeline. 
- Training Pipeline.
- Inference Pipeline.


## Feature Engineering Pipeline
We used two different data sources: Yahoo Finance API to get historical and live stock market prices for three different compamniess Apple, Amazon and Meta, and we scrapped the economic news related to those three companies from Investing.conm using BeautifulSoup library.
the pipeline retrieve the data from both sources, compute the sentiment analysis score for each news article using Vader analyzer and store them on hopsworks first. Next, the feature view notebook upload this data from hopsworks and aggregate the news dataset based on the date and merge the two datasets and push them again to hopsworks as a feature group and create a training data.

## Training Pipeline
In order to predict the future stock market price, we created three different models one for each company. This pipeline retrieve first the training data, refactors it into three different datasets, create a model for each and fit the data. At the end we test the model on a subset of unseen data to control the rmse error on the testing data.

### Model Structure
In order to perform the prediction, we used LSTM model. The architecture we used is as follows:

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 60, 100)           42400     
                                                                 
 lstm_1 (LSTM)               (None, 100)               80400     
                                                                 
 dense (Dense)               (None, 25)                2525      
                                                                 
 dense_1 (Dense)             (None, 1)                 26        
                                                                 
=================================================================
Total params: 125,351
Trainable params: 125,351
Non-trainable params: 0


## Inference Pipeline
