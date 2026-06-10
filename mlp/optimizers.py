import numpy as np

class Optimizer:
    """
    Interface base para todos os otimizadores.
    """
    def update(self, params: list, grads: list) -> None:
        """
        Atualiza os parâmetros in-place usando os gradientes fornecidos.
        
        Parâmetros:
            params (list): Lista de arrays NumPy contendo os parâmetros [W1, b1, W2, b2, ...].
            grads (list): Lista de arrays NumPy contendo os gradientes correspondentes [dW1, db1, dW2, db2, ...].
        """
        raise NotImplementedError

class SGD(Optimizer):
    """
    Otimizador Stochastic Gradient Descent (SGD) clássico.
    Fórmula de atualização: theta = theta - lr * grad
    """
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def update(self, params: list, grads: list) -> None:
        for param, grad in zip(params, grads):
            # Modificação in-place
            param -= self.lr * grad

class SGDMomentum(Optimizer):
    """
    Otimizador SGD com Momentum. Suaviza oscilações e acelera a convergência.
    Fórmula:
        v = beta * v + (1 - beta) * grad
        theta = theta - lr * v
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.beta = momentum
        self.velocities = None  # Inicializado na primeira chamada do update

    def update(self, params: list, grads: list) -> None:
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        for v, param, grad in zip(self.velocities, params, grads):
            # Atualiza a velocidade acumulada (in-place)
            v[:] = self.beta * v + (1 - self.beta) * grad
            # Atualiza o parâmetro
            param -= self.lr * v


def _test_optimizers() -> None:
    params = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.0], [0.0]])]
    grads = [np.ones_like(params[0]), np.ones_like(params[1])]

    sgd = SGD(learning_rate=0.1)
    sgd.update(params, grads)
    print("SGD params:")
    print(params)

    momentum = SGDMomentum(learning_rate=0.1, momentum=0.9)
    params = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.0], [0.0]])]
    momentum.update(params, grads)
    print("SGD with Momentum params:")
    print(params)


if __name__ == "__main__":
    _test_optimizers()
