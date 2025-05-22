import numpy as np

# Mean Square Error
def mse(y_true,y_pred):
    true = np.array(y_true)
    pred = np.array(y_pred)
    
    return np.mean((true-pred)**2)
    

# Mean Absolute Error
def mae(y_true,y_pred):
    true = np.array(y_true)
    pred = np.array(y_pred)
    
    return np.mean(np.abs(true-pred))
    
# Binary Cross Entropy
def bce(y_true,y_pred,epsilon=1e-12):
    true = np.array(y_true)
    pred = np.clip(np.array(y_pred), epsilon, 1 - epsilon) # Avoid log(0)
    return -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred)) 


print(mse([0, 1, 0], [0.1, 0.8, 0.1]))
print(mae([0, 1, 0], [0.1, 0.8, 0.1]))
print(bce([0, 1, 0], [0.1, 0.8, 0.1]))