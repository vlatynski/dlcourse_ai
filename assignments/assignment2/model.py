import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        reg, float - L2 regularization strength
        """
        print("n_input", n_input)
        print("n_output", n_output)
        print("hidden_layer_size", hidden_layer_size)
        self.reg = reg
        self.nn_layers = []
        # TODO Create necessary layers
        #for layer in range(hidden_layer_size):
        self.nn_layers.append(FullyConnectedLayer(n_input, hidden_layer_size))
        self.nn_layers.append(ReLULayer())
        self.nn_layers.append(FullyConnectedLayer(hidden_layer_size, n_output))

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: use self.params()
        X_next = X.copy()
        #print("----FORWARD PATH-----")
        #print("X in shape", X.shape)
        for layer in self.nn_layers:
            X_next = layer.forward(X_next)

        #print("----BACKWARD PATH-----")
        loss, grad = softmax_with_cross_entropy(X_next, y)

        l2 = 0.
        #print("GRAD out:", grad.shape)
        for layer in reversed(self.nn_layers):
            grad = layer.backward(grad)
            grad_l2 = 0
            for params in layer.params():
                param = layer.params()[params]
                loss_d, grad_d = l2_regularization(param.value, self.reg)
                param.grad += grad_d
                l2 += loss_d
            grad += grad_l2

        loss += l2

        #print("--------- END -----")
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        #X_next = X.copy()
        for layer in self.nn_layers:
            X = layer.forward(X)
        pred = np.argmax(X, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        for layer_num in range(len(self.nn_layers)):
            for i in self.nn_layers[layer_num].params():
                result[str(layer_num) + "_" + i] = self.nn_layers[layer_num].params()[i]

        return result

