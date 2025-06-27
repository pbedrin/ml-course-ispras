import numpy as np
import time


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_generator = np.random.RandomState(random_seed)

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector for initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        if trace:
            history = {"time": [], "func": [], "func_val": []}
        self.w = w_0 if w_0 is not None else self.random_generator.rand(X.shape[1])
        
        if self.batch_size:
            num_objects = y.shape[0]
            num_batches = num_objects // self.batch_size
            self.shuffled_objects_indices = np.arange(num_objects)

        for epoch in range(self.max_iter):
            start_time = time.time()
            w_prev = self.w.copy()

            if self.batch_size:
                self.random_generator.shuffle(self.shuffled_objects_indices)
                for batch_idx in range(num_batches):
                    batch_indices = self.shuffled_objects_indices[
                        batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                    ]
                    grad = self.loss_function.grad(X[batch_indices], y[batch_indices], self.w)
                    lr = self.step_alpha / (epoch * num_batches + batch_idx + 1) ** self.step_beta
                    self.w -= lr * grad
            else:
                grad = self.loss_function.grad(X, y, self.w)
                lr = self.step_alpha / (epoch + 1) ** self.step_beta
                self.w -= lr * grad

            if trace:
                history["time"].append(time.time() - start_time)
                history["func"].append(self.loss_function.func(X, y, self.w))
                if X_val is not None and y_val is not None:
                    history["func_val"].append(self.loss_function.func(X_val, y_val, self.w))
            
            if np.linalg.norm(w_prev - self.w) < self.tolerance:
                break

        if trace:
            return history

    def predict(self, X):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        return X @ self.w

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d model weights vector.
        """
        return self.w

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.func(X, y, self.w)
