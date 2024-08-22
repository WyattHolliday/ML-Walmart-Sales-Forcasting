# ML-Walmart-Sales-Forcasting
## Overview
A project by Wyatt Holliday to use machine learning to predict Walmart weekly sales. This project consist of three seperate machine learning methods a neural network, a random forest decision tree, and a regression support vector machine. Each of which I used k-fold-cross-validation with grid search to tune the hyperparameters. All of the predictions outputed by these are then put together in an ensemble using a weighted average so that the new prediction is more accurate then any of the individuals.

The python libraries used are torch, sklearn, numpy, pandas, joblib, and timeit.

## Data
The original data stored in dataset/Walmart_Data_Analysis_and_Forcasting.csv has the features Store Location, Date, Weekly_Sales, Whether it is a Holiday, Temperature, Fuel_Price, CPI, and Unemployment. The data is further featured engineered using one-hot encoding to better interpret categorical variables so that whether or not it is each of the 12 possible months, 31 days and 45 store locations each are their own feature. Now removing the original Store Location and Date features there is a total of 93 features in this dataset. Lastly the non-binary features are normalized using StandardScaler/feature distribution z-score.

Data found on kaggle at https://www.kaggle.com/datasets/asahu40/walmart-data-analysis-and-forcasting.

## Error Functions
For this project I used three error functions, Mean Squared Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). I used MSE as a loss function for my nueral network as I found it worked well. I used MAPE to make judgements on the models and determmine if they were overfitting as I found it easier to read and especially visually parse. I used MAPE to put my results in the context of the original data and get a understanding of how close to the labels the predictions were relative to thir size.

## Neural Network
I created the neural network using pytorch. I used mean squared error as my loss function and the ending hyperparameters were batch_size: 512, lr: 0.01, weight_decay: 1e-5, max_patience: 500, and epochs: 15000.

The final model, 'models/nn_model.pth' had an end result of a 3.97% MAPE on the testing data. This means that on average it is 3.97% off in the positive or negative direction of the actual sales data.

To prevent overfitting I added L2 regularization/weight_decay and dropout layers, also to better generalize the network. I added batch normalization to increase the stability, decrease overfitting and to be able to speed up the learning rate. I added early stopping with patience to ensure that the ending network had as low a validation error as possible. A patience of 500 meaning that after 500 epochs of no validation error inprovment early stopping would commence and the model at the lowest validation error would be remembered. I did this to try to prevent the model from overfitting to the validation data as much as it was.

Here is an example of the loss curve (Using MSE loss)
![walmart_nn_loss](https://github.com/user-attachments/assets/8f9d2113-1e05-4e6c-afda-a2b5aa45bc56)

Here is the training (blue) and validation (orange) error Using (MAE)
![walmart_nn_mae](https://github.com/user-attachments/assets/343422c8-7d57-413f-add5-529ec74a1c6f)

## Random Forest
I used sklearn.ensemble.RandomForestRegressor as the base for my model and tuned the hyperparameters. The ending hyperparameters were max_depth: 70, min_samples_split: 12, min_samples_leaf: 2, max_features: 0.65, and max_leaf_nodes: 1000.

The final model, 'models/random_forest_model.pkl' had an end result of a 5.96% MAPE. So, on average it is 5.96% off in the positive or negative direction of the actual sales data.

## Support Vector Regression
I used sklearn.svm.SVR as the base for my model and tuned the hyperparameters. The ending hyperparameters were kernel: 'poly', C: 1e5, gamma: 0.1, coef0: 2.25, and degree: 3.

The final model, 'models/svm_model.pkl' had an end result of a 5.66% MAPE. So, on average it is 5.66% off in the positive or negative direction of the actual sales data.

## The Ensemble
### Total Ensemble
To get the most accurate predictions possible I took a weighted average of the predictions of all three models for a given label. To find what each model should be weighted as I parameter tuned used grid search again. The weights I found were SVR: 0.09 RandomForestRegressor: 0.05 NeuralNet: 0.86. The predictions output my this ensemble had a MAPE of 3.88% on the test data. This is a 2% decrease in error from the NeuralNet model, 35% from the Random Forest model, and 31% from the Support Vector model.

### Random Forest, Support Vector Regression
Notably the nueral network was far more accurate then the other models making it extremely highly weighted, because of this the benefits of using an ensemble seem not as apperent as they could be. So I also decided to create an ensemble with just the simularly accurate random forest and support vector models. The weights I found for this were surprisingly SVR: 0.5 RandomForestRegressor: 0.5. The predictions output my this ensemble had a MAPE of 5.12% on the test data. This is a 14% decrease in error from the Random Forest model, and 9.5% from the Support Vector model. Much more substantial improvements in the error.

## Future Improvements
I think the most obvious way to improve the overall model would be to change the way that the ensemble is calculated. Rather than having every single prediction by a model be weighted the same weight have the weights be more complex and trained to find the relationship between all of the predictions and the label. To do this I would use stacking and create a meta-model (likely a Neural Network) and train it with the normalized results from the other models being the inputs and output being the label prediction.