import numpy as np
import math

ZERO = 0
ONE = 1
# create a neural network that can learn XOR logic

# N1 -> N3 -> N5
# N2 -> N4 -> N5
LOOPCOUNT = 100_000
LEARNING_RATE = 0.2

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

def CalculateErrorLastLayer(Neuron, target):
    return Neuron * (1 - Neuron) * (target - Neuron)

def CalculateErrorHiddenLayer(Neuron, Weight, backPropVal):
    return Neuron * (1 - Neuron) * (Weight * backPropVal)

def UpdateWeight(neuronError, neuron):
    return LEARNING_RATE * neuronError * neuron

def UpdateBias(neuronError, bias):
    return LEARNING_RATE * neuronError * bias

# Train the neural network
def TrainAi():
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
    for j in range(LOOPCOUNT):
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
            if j == LOOPCOUNT - 1:
                print("Input:", input0, input1, end=" | ", sep=" ")
                print("TARGET:", target, end=" | ", sep=" ")
                print(Neuron5)

            error = target - Neuron5
            # back propagation
            Neuron5Error = CalculateErrorLastLayer(Neuron5, target)
            Neuron3Error = CalculateErrorHiddenLayer(Neuron3, weight5, Neuron5Error)
            Neuron4Error = CalculateErrorHiddenLayer(Neuron4, weight6, Neuron5Error)

            # update weights
            weight6 += UpdateWeight(Neuron5Error, Neuron4)
            weight5 += UpdateWeight(Neuron5Error, Neuron3)
            weight4 += UpdateWeight(Neuron4Error, input1)
            weight3 += UpdateWeight(Neuron4Error, input0)
            weight2 += UpdateWeight(Neuron3Error, input1)
            weight1 += UpdateWeight(Neuron3Error, input0)
            # update biases
            bias3 -= UpdateBias(Neuron5Error, bias3)
            bias2 -= UpdateBias(Neuron4Error, bias2)
            bias1 -= UpdateBias(Neuron3Error, bias1)
TrainAi()