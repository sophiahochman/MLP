import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU(z) = max(0, z)
    """
    return np.maximum(0, z)

def relu_backward(z: np.ndarray) -> np.ndarray:
    """
    Derivada da ReLU em relação a z.
    Retorna 1.0 onde z > 0 e 0.0 caso contrário.
    """
    return (z > 0).astype(float)

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax numericamente estável.
    Assume que cada coluna de z representa uma amostra e cada linha representa uma classe.
    """
    # Subtrai o máximo de cada coluna para estabilidade numérica (axis=0)
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Função Sigmoid estável.
    """
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def sigmoid_backward(z: np.ndarray) -> np.ndarray:
    """
    Derivada da Sigmoid.
    """
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh(z: np.ndarray) -> np.ndarray:
    """
    Função Tangente Hiperbólica.
    """
    return np.tanh(z)

def tanh_backward(z: np.ndarray) -> np.ndarray:
    """
    Derivada da Tangente Hiperbólica.
    """
    return 1.0 - np.tanh(z) ** 2

# Registro de ativações para uso dinâmico na classe MLP
ACTIVATIONS = {
    "relu":    (relu,    relu_backward),
    "sigmoid": (sigmoid, sigmoid_backward),
    "tanh":    (tanh,    tanh_backward),
}


def _test_activations() -> None:
    z = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.0]])
    print("z:")
    print(z)
    print("relu(z):")
    print(relu(z))
    print("sigmoid(z):")
    print(sigmoid(z))
    print("tanh(z):")
    print(tanh(z))
    print("softmax(z):")
    print(softmax(z))


if __name__ == "__main__":
    _test_activations()
