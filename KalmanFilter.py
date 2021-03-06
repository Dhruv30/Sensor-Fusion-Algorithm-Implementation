import numpy as np

# Actual data of the object, each term represents the data at 1 arbitrary unit of time (Ex. 1 minute)
positionData = np.array([4000, 4260, 4550, 4860, 5110])
velocityData = np.array([280, 282, 285, 286, 290])

# The time interval between each of the terms in the above arrays (set to 1 for simplicity)
time = 1;

# Represents the initial error level designated to the position and velocity (used for the calculation of the initial covariance matrix)
positionErrorInit = 20
velocityErrorInit = 5

# Represents the actual error level designated to the position and velocity (used for calculating later covariance matrices)
positionErrorActual = 21
velocityErrorActual = 6


def statePrediction(position, velocity, time):
    dynamicMatrix = np.array([[1, time],
                                 [0, 1]])
    state = np.array([[position],
                             [velocity]])
    
    # Predicts the next state of the object based on its velocity and position
    newState = np.matmul(dynamicMatrix, state)
    
    return newState

                         
def getCovariance(sigma1, sigma2): # Source: https://gist.github.com/jaems33/6f445734f777db08d90892681641eb2e#file-kalman_filter_tracking_plane-py
    cov1_2 = sigma1 * sigma2
    cov2_1 = sigma2 * sigma1
    cov_matrix = np.array([[sigma1 ** 2, cov1_2],
                           [cov2_1, sigma2 ** 2]])
    return np.diag(np.diag(cov_matrix))
                         
                       
# Initial Covariance Matrix:
covarianceMatrix = getCovariance(positionErrorInit, velocityErrorInit)
dynamicMatrix = np.array([[1, time],
              [0, 1]])
state = np.array([[positionData[0]],[velocityData[0]]])

index1 = 1;

while(index1 < positionData.size):
    # Calculating new state and new covariance matrix
    state = statePrediction(state[0][0], state[1][0], 1)
    covarianceMatrix = dynamicMatrix.dot(covarianceMatrix).dot(dynamicMatrix.T)
    
    # Calculating Kalman Gain
    noiseCovariance = getCovariance(positionErrorActual, velocityErrorActual)
    newCovariance = covarianceMatrix + noiseCovariance 
    kalmanGain = covarianceMatrix.dot(np.linalg.inv(newCovariance))
    
    stateData = np.array([[positionData[index1]],[velocityData[index1]]])
    
    # Your new state is:
    state = state + kalmanGain.dot(stateData - state)
    
    # Your new covariance matrix is:
    covarianceMatrix = (np.identity(2) - kalmanGain).dot(covarianceMatrix)
    
    index1 = index1 + 1
    
    
print(state)