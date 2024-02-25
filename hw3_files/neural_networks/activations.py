"""
Author: Sophia Sanborn
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: https://github.com/sophiaas
"""

import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """Abstract class defining the common interface for all activation methods."""

    def __call__(self, Z):
        return self.forward(Z)

    @abstractmethod
    def forward(self, Z):
        pass


def initialize_activation(name: str) -> Activation:
    """Factory method to return an Activation object of the specified type."""
    if name == "identity":
        return Identity()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return TanH()
    elif name == "relu":
        return ReLU()
    elif name == "softmax":
        return SoftMax()
    else:
        raise NotImplementedError("{} activation is not implemented".format(name))


class Identity(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = z.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = z.
        
        Parameters
        ----------
        Z   input to the `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            (same shape as `Z`)

        Returns
        -------
        derivative of loss w.r.t. 'Z'
        """
        return dY


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function:
        f(z) = 1 / (1 + exp(-z))
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        return 1 / (1 + np.exp(-Z))

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid.
        
        Parameters
        ----------
        Z   input to the `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            (same shape as `Z`)

        Returns
        -------
        derivative of loss w.r.t. 'Z'
        """
        ### YOUR CODE HERE ###
        fn = self.forward(Z)
        return dY * (-fn ** 2 + fn)


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = tanh(z).
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return 2 / (1 + np.exp(-2 * Z)) - 1

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = tanh(z).
        
        Parameters
        ----------
        Z   input to the `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            (same shape as 'Z')

        Returns
        -------
        derivative of loss w.r.t. 'Z'
        """
        fn = self.forward(Z)
        return dY * (1 - fn ** 2)  # it seems to be dY * -\nabla_{f}(Z).


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(z) = z if z >= 0
               0 otherwise
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        return Z * (Z >= 0)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.
        
        Parameters
        ----------
        Z   input to the `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            (same shape as `Z`)

        Returns
        -------
        derivative of loss w.r.t. 'Z'
        """
        ### YOUR CODE HERE ###
        return np.ones(Z.shape) * (Z >= 0) * dY


class SoftMax(Activation):
    def __init__(self):
        super().__init__()
    
    def forwardOnce(self, Z: np.ndarray) -> np.ndarray:
        """ return the forward pass of one train sample. """
        maxOfZ = np.max(Z)
        expOfZ = np.exp(Z - maxOfZ)
        sumOfZ = np.sum(expOfZ)
        return expOfZ / sumOfZ

    def backwardOnce(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        fn = self.forward(Z)
        jacobi = np.diag(fn.reshape(fn.shape[1])) - np.dot(fn.T, fn)
        return dY @ jacobi

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for the softmax activation.
        Hint: The naive implementation might not be numerically stable. (Why?)
        
        Parameters
        ----------
        Z  input pre-activations (batch size, num_activations)

        Returns
        -------
        f(z), which is the array resulting from applying the softmax function
        to each sample's activations (same shape as 'Z')
        """
        ### YOUR CODE HERE ###
        # note: due to numerical stability, often works like this:
        res = np.zeros(Z.shape)
        for r in range(Z.shape[0]):
            row = Z[r]
            row = row.reshape((1, row.shape[0]))
            res[r] = self.forwardOnce(row)
        return res
    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for softmax activation.
        
        Parameters
        ----------
        Z   input to the `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. Z
        """
        ### YOUR CODE HERE ###
        res = np.zeros(Z.shape)
        for r in range(Z.shape[0]):
            row = Z[r]
            row = row.reshape((1, row.shape[0]))
            diff = dY[r]
            diff = diff.reshape((1, diff.shape[0]))
            back = self.backwardOnce(row, diff)
            # print(back.shape)
            res[r] = back
        return res
