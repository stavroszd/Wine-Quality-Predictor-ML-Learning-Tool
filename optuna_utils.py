#%%Package imports
from torch_utils import NN_train_test_n_times

#%%Optuna objective function
def objective(trial, model, X, y, task, output_shape = 3, input_shape = 11):
    #Suggest values for hyper - parameters 
    # Suggest values for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)  # Log scale for LR
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)  # Regularization
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])  # Choose batch size

    #Call my training and testing function to get the test metric
    #Optuna with optimize it in regards to this. This is something we define in our methodology.

    test_metric = NN_train_test_n_times(epochs = 250, n = 1, X = X, y = y, model_class = model, task = task, batch_size = batch_size, learning_rate = learning_rate, weight_decay = weight_decay, output_shape = output_shape, input_shape = input_shape)

    return test_metric #This is what we want to optimize