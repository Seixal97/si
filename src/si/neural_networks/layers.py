import copy
from abc import abstractmethod

import numpy as np

from si.neural_networks.optimizers import Optimizer


class Layer:
    """
    Base class for neural network layers.
    """

    @abstractmethod
    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input, i.e., computes the output of a layer for a given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error: float) -> float:
        """
        Perform backward propagation on the given output error, i.e., computes dE/dX for a given dE/dY and update
        parameters if any.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        raise NotImplementedError

    def layer_name(self) -> str:
        """
        Returns the name of the layer.

        Returns
        -------
        str
            The name of the layer.
        """
        return self.__class__.__name__

    @abstractmethod
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        raise NotImplementedError

    def set_input_shape(self, shape: tuple):
        """
        Sets the shape of the input to the layer.

        Parameters
        ----------
        shape: tuple
            The shape of the input to the layer.
        """
        self._input_shape = shape

    def input_shape(self) -> tuple:
        """
        Returns the shape of the input to the layer.

        Returns
        -------
        tuple
            The shape of the input to the layer.
        """
        return self._input_shape

    @abstractmethod
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        raise NotImplementedError


class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        # [[w f1n1, w f1n2, w f1n3, ...],
        # [w f2n1, w f2n2, w f2n3, ...], 
        # [w f3n1, w f3n2, w f3n3, ...],...]
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5

        # initialize biases to 0
        # [[b1, b2, b3, ...]]
        self.biases = np.zeros((1, self.n_units))

        # initialize optimizer 
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        # returns the number of parameters of the layer
        # (n_input_features * n_units) + biases
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        # computes the layer output
        # [[f1, f2, f3, ...]]
        self.input = input

        # [[f1, f2, f3, ...]] * [[w f1n1, w f1n2, w f1n3, ...] + [[b1, b2, b3, ...]]
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        input_error = np.dot(output_error, self.weights.T)
        # computes the weight error: dE/dW = X.T * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # computes the bias error: dE/dB = dE/dY
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error

    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,)
    

class Dropout(Layer):
    '''
    Dropout layer of a neural network. A random set of neurons is temporarily ignored dropped out) during training, 
    helping prevent overfitting by promotingrobustness and generalization in the model
    '''

    def __init__(self, probability: float):
        '''
        Initialize the dropout layer.

        Parameters
        ----------
        probability: float
            The probability of dropping a neuron between 0 and 1 (dropout rate).

        Attributes
        ----------
        mask: numpy.ndarray
            binomial mask that sets some inputs to 0 based on the probability
        input: numpy.ndarray
            the input to the layer
        output: numpy.ndarray
            the output of the layer

        '''
        super().__init__()

        # means the percentage of neurons to drop
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        '''
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        '''
        self.input = input

        #if we are in training mode (we only apply dropout during training)
        if training:

            #compute the scaling factor to apply at test time
            scaling_factor = 1 / (1 - self.probability)

            #compute the mask
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)

            #compute the output
            self.output = input * self.mask * scaling_factor

            #return the output
            return self.output
        
        #if we are in inference mode
        else:
            self.output = input
            return self.output
        
    def backward_propagation(self, output_error: np.ndarray) -> float:
        '''
        Perform backward propagation on the given output error.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        '''
        #compute the input error
        input_error = output_error * self.mask

        #return the input error
        return input_error
    
    def output_shape(self) -> tuple:
        '''
        Returns the shape of the output of the layer.
        '''
        return self.input_shape()
    
    def parameters(self) -> int:
        '''
        Returns the number of parameters of the layer.
        '''
        # we do this so that the dropout layer can be used in a model
        return 0
        
if __name__ == "__main__":
    #test the dropout layer
    np.random.seed(42)
    x = np.array([[5, 8, 9, 5, 10, 2, 1, 7, 6, 9]])
    print("x:", x)
    dropout = Dropout(0.5)
    print("output training:", dropout.forward_propagation(x, training=True))
    print("mask:", dropout.mask)
    
    print("output inference:", dropout.forward_propagation(x, training=False))

    
