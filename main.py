import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib
import timeit

def load_data():
    traindata = pd.read_csv('dataset/Walmart_Data_Analysis_and_Forcasting.csv')

    traindata['Date'] = pd.to_datetime(traindata['Date'], format='%d-%m-%Y', dayfirst=True)

    # One-hot encode the month, day and store columns
    month_dummies = pd.get_dummies(traindata['Date'].dt.month, prefix='Is_Month')
    day_dummies = pd.get_dummies(traindata['Date'].dt.day, prefix='Is_Day')
    store_dummies = pd.get_dummies(traindata['Store'], prefix='Is_Store')

    # Concatenate the dummy columns with the original DataFrame
    traindata = pd.concat([traindata, month_dummies, day_dummies, store_dummies], axis=1)

    # Drop the engineered columns
    traindata = traindata.drop(columns=['Date', 'Store'])

    data_y = traindata['Weekly_Sales'].astype(np.float32)
    data_X = traindata.drop(columns=['Weekly_Sales']).astype(np.float32)

    # Identify binary and continuous variables
    binary_cols = data_X.columns[data_X.nunique() == 2]  # Columns with exactly two unique values
    continuous_cols = data_X.columns.difference(binary_cols)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    train_X = scaler.fit_transform(data_X[continuous_cols])

    train_X = np.hstack((train_X, data_X[binary_cols].values))

    train_y = data_y.to_numpy()

    
    # Deterministically shuffle and split off data for testing
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.1, random_state=42)

    return train_X, train_y , test_X, test_y

def k_fold_cross_validation(k, X, y, func, params=None):
    # print(f'Performing {k}-fold cross validation...')
    kf = KFold(n_splits=k)
    lowest_val_mae = np.inf
    val_maes = []
    val_mapes = []
    train_maes = []
    final_model = None
    fold = 1

    for train_index, val_index in kf.split(X):
        # Get k fold train and validation sets
        train_X, val_X = X[train_index], X[val_index]
        train_y, val_y = y[train_index], y[val_index]

        # Train the model
        train_mae, val_mae, val_mape, model = func(train_X, train_y, val_X=val_X, val_y=val_y, k_fold=True, params=params)
        if isinstance(model, NeuralNet):
            print(f'Fold {fold} model at lowest val, Train MAE: {int(train_mae)}, Val MAE: {int(val_mae)}, Val MAPE: {val_mape}%')

        # Save the model with the lowest validation MAE
        if val_mae < lowest_val_mae:
            lowest_val_mae = val_mae
            final_model = model

        val_maes.append(val_mae)
        val_mapes.append(val_mape)
        train_maes.append(train_mae)

        fold += 1
    
    if isinstance(model, RandomForestRegressor) and round(np.mean(val_mapes), 2) <= 7:
        max_depth = params["max_depth"]
        min_split = params["min_split"]
        min_leaf = params["min_leaf"]
        max_feature = params["max_features"]
        max_leaf = params["max_leaf"]
        print(f'Random Forest, Max Depth: {max_depth}, Min Split: {min_split}, Min Leaf: {min_leaf}, Max Features: {max_feature}, Max Leaf: {max_leaf}')
        print(f'Average Train MAE: {int(np.mean(train_maes))}, Average Val MAE: {int(np.mean(val_maes))}, Average Val MAPE: {round(np.mean(val_mapes), 2)}%, MAE difference: {int(np.mean(val_maes)) - int(np.mean(train_maes))}')
        print()
        joblib.dump(model, 'models/trained_random_forest.pkl')
    elif isinstance(model, NeuralNet):
        print(f'Average Train MAE: {int(np.mean(train_maes))}, Average Val MAE: {int(np.mean(val_maes))}, Average Val MAPE: {round(np.mean(val_mapes), 2)}%, MAE difference: {int(np.mean(val_maes)) - int(np.mean(train_maes))}')
    elif isinstance(model, SVR) and round(np.mean(val_mapes), 2) <= 7:
        print(f'SVR, kernel: {params["kernel"]} C: {params["C"]} gamma: {params["gamma"]} coef0: {params["coef0"]} degree: {params["degree"]}')
        print(f'Average Train MAE: {int(np.mean(train_maes))}, Average Val MAE: {int(np.mean(val_maes))}, Average Val MAPE: {round(np.mean(val_mapes), 2)}%, MAE difference: {int(np.mean(val_maes)) - int(np.mean(train_maes))}')
        print()
        joblib.dump(model, 'models/trained_svr.pkl')

    return final_model

def train_nn(X, y, model=None, epochs=15000, test_size=0.2, val_X=None, val_y=None, deterministic=True, k_fold=False, params=None):
    if val_X is None and val_y is None: # if no validation set is provided, split the data
        if deterministic:
            train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            print('------------------------------ non deterministic data split ------------------------------')
            train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=test_size)
    else:
        train_X, train_y = X, y

    if params is None:
        params = {
            "batch_size": 512,
            "lr": 0.01,
            "weight_decay": 1e-5,
            "max_patience": 500
        }
    
    if model is None:
        model = NeuralNet(lr=params["lr"], weight_decay=params["weight_decay"])

    train_dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    best_model = None
    best_epoch = -1
    best_val_mae = np.inf
    best_val_mape = -1
    train_mae_for_best = -1
    patience = 0
    max_patience = params["max_patience"]

    losses = []
    train_maes = []
    val_maes = []
    start_time = timeit.default_timer()
    for epoch in range(epochs):
        train_batch_maes = []

        model.train()
        for batch_X, batch_y in train_loader: # train model batch by batch
            batch_y = batch_y.unsqueeze(1)
            model.optimizer.zero_grad()
            output = model(batch_X)
            loss = model.loss_fn(output, batch_y)
            loss.backward()
            model.optimizer.step()

            train_batch_maes.append(np.mean(np.abs(output.detach().numpy() - batch_y.numpy())))
        # Calculate training metrics
        train_mae = np.mean(train_batch_maes)
        train_maes.append(train_mae)
        losses.append(loss.item())
        
        val_loss = 0
        val_predictions = []
        val_targets = []

        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in val_loader: # validate model batch by batch
                batch_y = batch_y.unsqueeze(1)
                predictions = model(batch_X)
                val_loss += model.loss_fn(predictions, batch_y).item()
                val_predictions.append(predictions.numpy())
                val_targets.append(batch_y.numpy())

        # Total lists for epoch
        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)

        # Calculate validation metrics
        val_mae = np.mean(np.abs(val_predictions - val_targets))
        val_maes.append(val_mae)
        val_mape = round(np.mean(np.abs(val_predictions - val_targets) / val_targets) * 100, 2)

        if not k_fold: # if not k fold, print the metrics
            print(f'Epoch: {epoch+1}/{epochs}, Training MAE {int(train_mae)}, Val MAE: {int(val_mae)}, Val MAPE: {val_mape}%')
        
        # Save the model with the lowest validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_mape = val_mape
            train_mae_for_best = train_mae
            best_model = model
            best_epoch = epoch+1
            patience = 0
        else: # Early stopping if validation MAE does not improve
            patience += 1
            if patience > max_patience:
                if not k_fold:
                    print('Early stopping...')
                break
    
    if not k_fold:
        print(f'-- Training time: {((timeit.default_timer() - start_time) / 60):.2f} minutes, seconds per epoch: {(timeit.default_timer() - start_time) / epochs:.2f}')
        torch.save(model.state_dict(), 'models/trained_nn.pth')
        print(f"Best model, Epoch: {best_epoch}, Train MAE: {int(train_mae_for_best)}, Val MAE: {int(best_val_mae)}, Val MAPE: {best_val_mape}%")
        plot(train_maes, 'Error', 'Epoch', 'Mean Absolute Error', val_maes, 'Train', 'Validation')
        plot(losses, 'Training Loss', 'Epoch', 'Loss')

    return train_mae_for_best, best_val_mae, best_val_mape, best_model

def test_nn(X, y, model):
    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_y = batch_y.unsqueeze(1)
            predictions = model(batch_X)
            test_predictions.append(predictions.numpy())
            test_targets.append(batch_y.numpy())
        
    test_predictions = np.concatenate(test_predictions)
    test_targets = np.concatenate(test_targets)
    test_mae = np.mean(np.abs(test_predictions - test_targets))
    test_mape = round(np.mean(np.abs(test_predictions - test_targets) / test_targets) * 100, 2)
    print(f'Test MAE: {int(test_mae)}, Test MAPE: {test_mape}%')
    return test_mae

class NeuralNet(nn.Module):
    def __init__(self, lr=0.01, weight_decay=1e-5):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(93, 256)
        self.fc1_drop = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2_drop = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_drop(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc2_drop(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.fc4(x)
        return x

def random_forest(X, y, test_size=0.2, val_X=None, val_y=None, k_fold=False, params=None):
    if val_X is None and val_y is None: # if no validation set is provided, split the data
        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        train_X, train_y = X, y

    # Default to tuned parameters
    if params is None:
        params = {
            "max_depth": 70,
            "min_split": 12,
            "min_leaf": 2,
            "max_features": 0.65,
            "max_leaf": 1000,
            "random_state": 42
        }

    model = RandomForestRegressor(
        n_estimators=100,      # Number of boosting stages
        max_depth=params["max_depth"],           # Maximum depth of individual trees
        min_samples_split=params["min_split"],   # Minimum number of samples required to split an internal node
        min_samples_leaf=params["min_leaf"],     # Minimum number of samples required to be at a leaf node
        max_features=params["max_features"],     # Number of features to consider when looking for the best split
        max_leaf_nodes=params["max_leaf"],       # Grow trees with max_leaf_nodes in best-first fashion
        random_state=params["random_state"]
    )
    model.fit(train_X, train_y)

    train_predictions = model.predict(train_X)
    val_predictions = model.predict(val_X)

    # Calculate the mean absolute error (MAE) and mean absolute percentage error (MAPE)
    train_mae = np.mean(np.abs(train_predictions - train_y))
    val_mae = np.mean(np.abs(val_predictions - val_y))
    val_mape = round(np.mean(np.abs(val_predictions - val_y) / val_y) * 100, 2)

    if k_fold == False:
        print(f'Train MAE: {int(train_mae)}, Val MAE: {int(val_mae)}, Val MAPE: {val_mape}%')
        joblib.dump(model, 'models/trained_random_forest.pkl')
    
    return train_mae, val_mae, val_mape, model

def test_sklearn_model(model, test_X, test_y): # Predict data then calc errors
    test_predictions = model.predict(test_X)
    test_mae = np.mean(np.abs(test_predictions - test_y))
    test_mape = round(np.mean(np.abs(test_predictions - test_y) / test_y) * 100, 2)
    print(f'Test {model.__class__.__name__} MAE: {int(test_mae)}, MAPE: {test_mape}%\n')
    return test_mae

def support_vector_regression(X, y, test_size=0.2, val_X=None, val_y=None, k_fold=False, params=None):
    if val_X is None and val_y is None: # if no validation set is provided, split the data
        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        train_X, train_y = X, y
    
    # Default to tuned parameters
    if params == None:
        params = {
            "kernel": 'poly',
            "C": 1e5,
            "gamma": 0.1,
            "coef0": 2.25,
            "degree": 3
        }
    
    # Make and train model
    if params["kernel"] == 'linear':
        model = SVR(kernel=params["kernel"], C=params["C"])
    elif params["kernel"] == 'rbf':
        model = SVR(kernel=params["kernel"], C=params["C"], gamma=params["gamma"])
    elif params["kernel"] == 'poly':
        model = SVR(kernel=params["kernel"], C=params["C"], gamma=params["gamma"], coef0=params["coef0"], degree=params["degree"])
    elif params["kernel"] == 'sigmoid':
        model = SVR(kernel=params["kernel"], C=params["C"], gamma=params["gamma"], coef0=params["coef0"])
    model.fit(train_X, train_y)

    # Make predictions on training and validation data
    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)

    # Calculate the mean absolute error (MAE) and mean absolute percentage error (MAPE)
    train_mae = np.mean(np.abs(train_pred - train_y))
    val_mae = np.mean(np.abs(val_pred - val_y))
    val_mape = round(np.mean(np.abs(val_pred - val_y) / val_y) * 100, 2)

    if k_fold == False:
        print(f'Train MAE: {int(train_mae)}, Val MAE: {int(val_mae)}, Val MAPE: {val_mape}%')
        joblib.dump(model, 'models/trained_svr.pkl')
    
    return train_mae, val_mae, val_mape, model

def ensemble(models, X, y):
    ensemble_predictions = np.zeros(len(y))

    if sum([weight for _, weight in models]) != 1: # Weights should sum to 1
        print(f'ERROR: Weights in {models} do not sum to 1\n')
        return
    
    for model, weight in models:  # for each model, predict the output and multiply by the weight
        if isinstance(model, RandomForestRegressor) or isinstance(model, SVR):  # if model comes from sklearn
            ensemble_predictions += model.predict(X) * weight
            print(f'{model.__class__.__name__} MAPE: {round(np.mean(np.abs(model.predict(X) - y) / y) * 100, 2)}')
            
        elif isinstance(model, NeuralNet):  # if model is a torch model
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            model.eval()
            with torch.no_grad():
                predictions = []
                for batch_X, _ in loader:
                    preds = model(batch_X).squeeze().numpy()
                    predictions.append(preds)
                print(f'NueralNet MAPE: {round(np.mean(np.abs(np.concatenate(predictions) - y) / y) * 100, 2)}')

                # Convert predictions to numpy array and weight them
                predictions = np.concatenate(predictions) * weight
                ensemble_predictions += predictions
                
    # Calculate the mean absolute error (MAE) and mean absolute percentage error (MAPE)
    ensemble_mae = np.mean(np.abs(ensemble_predictions - y))
    ensemble_mape = round(np.mean(np.abs(ensemble_predictions - y) / y) * 100, 2)
    print(f'Ensemble MAE: {int(ensemble_mae)}, MAPE: {ensemble_mape}%\n')
    return ensemble_mae

def plot(data1, title, xlabel, ylabel, data2=None, label1=None, label2=None):
    if data2 is not None and label1 is not None and label2 is not None:
        plt.plot(data1, label=label1)
        plt.plot(data2, label=label2)
    else:
        plt.plot(data1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def main():
    train_X, train_y, test_X, test_y = load_data()

    nn_model = NeuralNet()
    nn_model.load_state_dict(torch.load('models/nn_model.pth'))
    rf_model = joblib.load("models/random_forest_model.pkl")
    svr_model = joblib.load("models/svr_model.pkl")

    svr_weights = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    rf_weights = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    for prop_rf_weight in rf_weights:
        for svr_weight in svr_weights:
            rf_weight = round((1 - svr_weight) * prop_rf_weight, 2)
            nn_weight = round((1 - svr_weight) * (1 - prop_rf_weight), 2)

            print(f'SVR Weight: {svr_weight}, RF Weight: {rf_weight}, NN Weight: {nn_weight}')
            ensemble_model_weights = [(svr_model, svr_weight), (rf_model, rf_weight), (nn_model, nn_weight)]
            ensemble(ensemble_model_weights, test_X, test_y)
            
    final_ensemble_model_weights = [(svr_model, 0.09), (rf_model, 0.05), (nn_model, 0.86)]

    # svr_weights = [0.0, 0.01, 0.02, 0.03, 0.04, 0.45, 0.05, 0.55, 0.06, 0.07, 0.08, 0.09, 0.1]
    svr_weights = [0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]
    for svr_weight in svr_weights:
        rf_weight = round(1 - svr_weight, 2)
        print(f'SVR Weight: {svr_weight}, RF Weight: {rf_weight}')
        ensemble_model_weights = [(svr_model, svr_weight), (rf_model, rf_weight)]
        ensemble(ensemble_model_weights, test_X, test_y)
    
    svr_rf_ensemble_model_weights = [(svr_model, 0.5), (rf_model, 0.5)]

if __name__ == '__main__':
    main()