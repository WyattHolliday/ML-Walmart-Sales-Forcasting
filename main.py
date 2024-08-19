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
from tqdm import tqdm
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
        # print(f'Fold {fold} model at lowest val, Train MAE: {int(train_mae)}, Val MAE: {int(val_mae)}, Val MAPE: {val_mape}%')

        # Save the model with the lowest validation MAE
        if val_mae < lowest_val_mae:
            lowest_val_mae = val_mae
            final_model = model

        val_maes.append(val_mae)
        val_mapes.append(val_mape)
        train_maes.append(train_mae)

        fold += 1
    
    if round(np.mean(val_mapes), 2) <= 7:
        max_depth = params["max_depth"]
        min_split = params["min_split"]
        min_leaf = params["min_leaf"]
        max_feature = params["max_features"]
        max_leaf = params["max_leaf"]
        print(f'Random Forest, Max Depth: {max_depth}, Min Split: {min_split}, Min Leaf: {min_leaf}, Max Features: {max_feature}, Max Leaf: {max_leaf}')
        print(f'Average Train MAE: {int(np.mean(train_maes))}, Average Val MAE: {int(np.mean(val_maes))}, Average Val MAPE: {round(np.mean(val_mapes), 2)}%, MAE difference: {int(np.mean(val_maes)) - int(np.mean(train_maes))}')
        print()

    return final_model

def train_nn(X, y, model=None, epochs=15000, batch_size=512, test_size=0.2, val_X=None, val_y=None, deterministic=True, k_fold=False, params=None):
    # print(f'Training Neural Network for {epochs} epochs with batch size {batch_size}...')
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
            "lr": 0.01,
            "weight_decay": 1e-5
        }
    
    if model is None:
        model = NeuralNet(lr=params["lr"], weight_decay=params["weight_decay"])

    train_dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_model = None
    best_epoch = -1
    best_val_mae = np.inf
    best_val_mape = -1
    train_mae_for_best = -1
    patience = 0
    max_patience = 500

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
        torch.save(model.state_dict(), 'models/trained_model.pth')
        print(f"Best model, Epoch: {best_epoch}, Train MAE: {int(train_mae_for_best)}, Val MAE: {int(best_val_mae)}, Val MAPE: {best_val_mape}%")
        plot(train_maes, 'Error', 'Epoch', 'Mean Absolute Error', val_maes, 'Train', 'Validation')
        plot(losses, 'Training Loss', 'Epoch', 'Loss')

    return train_mae_for_best, best_val_mae, best_val_mape, best_model

def test_nn(X, y, model):
    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

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
    if val_X is None and val_y is None:
        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        train_X, train_y = X, y

    if params is None:
        params = {
            "max_depth": 10,
            "min_split": 2,
            "min_leaf": 1,
            "max_features": 'sqrt',
            "max_leaf": None
        }

    model = RandomForestRegressor(
        n_estimators=100,      # Number of boosting stages
        max_depth=params["max_depth"],           # Maximum depth of individual trees
        min_samples_split=params["min_split"],   # Minimum number of samples required to split an internal node
        min_samples_leaf=params["min_leaf"],    # Minimum number of samples required to be at a leaf node
        max_features=params["max_features"],   # Number of features to consider when looking for the best split
        max_leaf_nodes=params["max_leaf"],   # Grow trees with max_leaf_nodes in best-first fashion
        random_state=42
    )
    model.fit(train_X, train_y)

    train_predictions = model.predict(train_X)
    val_predictions = model.predict(val_X)

    train_mae = np.mean(np.abs(train_predictions - train_y))
    val_mae = np.mean(np.abs(val_predictions - val_y))
    val_mape = round(np.mean(np.abs(val_predictions - val_y) / val_y) * 100, 2)

    if k_fold == False:
        print(f'Gradient Boosting')
        print(f'Train MAE: {int(train_mae)}, Val MAE: {int(val_mae)}, Val MAPE: {val_mape}%')
    
    return train_mae, val_mae, val_mape, model

def test_RF(model, test_X, test_y):
    test_predictions = model.predict(test_X)
    test_mae = np.mean(np.abs(test_predictions - test_y))
    test_mape = round(np.mean(np.abs(test_predictions - test_y) / test_y) * 100, 2)
    print(f'Test Random Forest MAE: {int(test_mae)}, MAPE: {test_mape}%')
    print()
    return test_mae

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
    # model = NeuralNet()
    # model.load_state_dict(torch.load('models/trained_model.pth'))
    # train_mae, val_mae, val_mape, model = train_nn(train_X, train_y, epochs=15000)

    # max_depth=params["max_depth"]
    # min_samples_split=params["min_split"]
    # min_samples_leaf=params["min_leaf"]
    # max_features=params["max_features"]
    # max_leaf_nodes=params["max_leaf"]

    # params = {}
    # max_depths = [93, 70, 50, 30]
    # min_splits = [2, 5, 10, 15]
    # min_leafs = [1, 2, 5, 10]
    # max_features = [1.0, 0.75, 0.5, 0.25]
    # max_leafs = [None, 1000, 500, 100]

    # for max_depth in max_depths:
    #     for min_split in min_splits:
    #         for min_leaf in min_leafs:
    #             for max_feature in max_features:
    #                 for max_leaf in max_leafs:
    #                     params["max_depth"] = max_depth
    #                     params["min_split"] = min_split
    #                     params["min_leaf"] = min_leaf
    #                     params["max_features"] = max_feature
    #                     params["max_leaf"] = max_leaf
    #                     k_fold_cross_validation(5, train_X, train_y, random_forest, params=params)

    # params = {}
    # max_depths = [40, 50, 60, 70, 80, 93]
    # min_splits = [6, 8, 10, 12, 14]
    # min_leafs = [2, 3, 4]
    # max_features = [0.65, 0.75, 0.85]
    # max_leafs = [None, 1000]
    # for max_depth in max_depths:
    #     for min_split in min_splits:
    #         for min_leaf in min_leafs:
    #             for max_feature in max_features:
    #                 for max_leaf in max_leafs:
    #                     params["max_depth"] = max_depth
    #                     params["min_split"] = min_split
    #                     params["min_leaf"] = min_leaf
    #                     params["max_features"] = max_feature
    #                     params["max_leaf"] = max_leaf
    #                     k_fold_cross_validation(5, train_X, train_y, random_forest, params=params)
    
    # model = k_fold_cross_validation(5, train_X, train_y, random_forest)

    # test_mae = test_nn(test_X, test_y, model)
    rf_params = {
        "max_depth": 70,
        "min_split": 12,
        "min_leaf": 2,
        "max_features": 0.65,
        "max_leaf": 1000
    }
    rf_model = k_fold_cross_validation(5, train_X, train_y, random_forest, params=rf_params)

if __name__ == '__main__':
    main()