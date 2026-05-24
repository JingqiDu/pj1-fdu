from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay


    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.matmul(X, self.params['W']) + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.matmul(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return np.matmul(grad, self.params['W'].T)

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.grads = {'W' : None, 'b' : None}
        self.params = {'W' : self.W, 'b' : self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None
        self.input_padded = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        self.input = X
        batch_size, _, H, W = X.shape
        K = self.kernel_size
        P = self.padding
        S = self.stride
        if P > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
        else:
            X_pad = X
        self.input_padded = X_pad
        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1
        out = np.zeros((batch_size, self.out_channels, H_out, W_out))

        for i in range(H_out):
            h_start = i * S
            h_end = h_start + K
            for j in range(W_out):
                w_start = j * S
                w_end = w_start + K
                window = X_pad[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.tensordot(window, self.params['W'], axes=([1, 2, 3], [1, 2, 3]))
        out += self.params['b']
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X_pad = self.input_padded
        batch_size, _, H_pad, W_pad = X_pad.shape
        K = self.kernel_size
        S = self.stride
        _, _, H_out, W_out = grads.shape
        dX_pad = np.zeros_like(X_pad)
        dW = np.zeros_like(self.params['W'])
        db = np.sum(grads, axis=(0, 2, 3), keepdims=True).reshape(1, self.out_channels, 1, 1)

        for i in range(H_out):
            h_start = i * S
            h_end = h_start + K
            for j in range(W_out):
                w_start = j * S
                w_end = w_start + K
                window = X_pad[:, :, h_start:h_end, w_start:w_end]
                grad_ij = grads[:, :, i, j]
                dW += np.tensordot(grad_ij, window, axes=([0], [0]))
                dX_pad[:, :, h_start:h_end, w_start:w_end] += np.tensordot(grad_ij, self.params['W'], axes=([1], [0]))

        self.grads['W'] = dW
        self.grads['b'] = db
        if self.padding > 0:
            return dX_pad[:, :, self.padding:H_pad-self.padding, self.padding:W_pad-self.padding]
        return dX_pad

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output

    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.labels = None
        self.predicts = None
        self.probs = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels.astype(np.int64)
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
        batch_indices = np.arange(labels.shape[0])
        correct_probs = self.probs[batch_indices, self.labels]
        return -np.mean(np.log(correct_probs + 1e-12))

    def backward(self):
        batch_size = self.labels.shape[0]
        one_hot = np.zeros((batch_size, self.max_classes))
        one_hot[np.arange(batch_size), self.labels] = 1
        if self.has_softmax:
            self.grads = (self.probs - one_hot) / batch_size
        else:
            self.grads = -one_hot / (self.probs + 1e-12) / batch_size
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self

class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
