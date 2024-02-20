from signal import siginterrupt
import numpy as np
import math
# import AITrainer

ZERO = 0
ONE = 1

# create a neural network that can learn XOR logic

# N1 -> N3 -> N5

# N2 -> N4 -> N5

# Weights
weight1 = np.random.rand() # goes to N3
weight2 = np.random.rand() # goes to N3
weight3 = np.random.rand() # goes to N4
weight4 = np.random.rand() # goes to N4
weight5 = np.random.rand() # goes to N5
weight6 = np.random.rand() # goes to N5

# Biases
bias1 = np.random.rand() # goes to N3
bias2 = np.random.rand() # goes to N4
bias3 = np.random.rand() # goes to N5

LEARNING_RATE = 0.1

# input data
INPUT = np.array([[0,0],[0,1],[1,0],[1,1]])

# output data
TARGET = np.array([0,1,1,0])

# call this function before Sigmoid to get num
def GetSigmoidInput(input1, input2, weight1, weight2, bias):
    return (input1 * weight1) + (input2 * weight2) - bias

# propgate forward
def Sigmoid(num):
    return 1 / (1 + math.exp(-num))

def CalculateError(Neuron):
    return Neuron * (1 - Neuron) * (TARGET - Neuron)

def UpdateWeight(Weight, Error, Bias):
    return Weight + LEARNING_RATE * -Bias * Error

def UpdateBias(Bias, Error):
    return Bias + LEARNING_RATE * -Bias * Error

# Train the neural network
def TrainAi():

    # for j in range(10000):
    for i in range(4):
        # forward propagation
        input0 = INPUT[i][ZERO]
        input1 = INPUT[i][ONE]
        target = TARGET[i]

        Neuron3 = GetSigmoidInput(input0, input1, weight1, weight2, bias1)
        Neuron4 = GetSigmoidInput(input0, input1, weight3, weight4, bias2)

        Neuron3 = Sigmoid(Neuron3)
        Neuron4 = Sigmoid(Neuron4)

        Neuron5 = GetSigmoidInput(Neuron3, Neuron4, weight5, weight6, bias3)
        Neuron5 = Sigmoid(Neuron5)

        # Print the results
        print("Input:", input0, input1, end=" | ", sep=" ")
        print("TARGET:", target, end=" | ", sep=" ")
        print(Neuron5)

        error = target - Neuron5
        # back propagation
        Neuron5Error = CalculateError(Neuron5)





TrainAi()