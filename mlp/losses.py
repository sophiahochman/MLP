import numpy as np

def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Converte um vetor de rótulos inteiros de formato (m,) em uma matriz one-hot de formato (n_classes, m).
    """
    m = y.shape[0]
    Y = np.zeros((n_classes, m), dtype=float)
    Y[y, np.arange(m)] = 1.0
    return Y

def cross_entropy_loss(A_out: np.ndarray, Y: np.ndarray) -> float:
    """
    Calcula a perda de Entropia Cruzada média sobre o mini-lote de dimensão (n_classes, m).
    """
    m = A_out.shape[1]
    # Clipping para evitar log(0)
    A_clipped = np.clip(A_out, 1e-15, 1.0)
    loss = -np.sum(Y * np.log(A_clipped)) / m
    return float(loss)

def softmax_crossentropy_backward(A_out: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Gradiente combinado da Softmax + Cross-Entropy em relação aos logits (dL/dZ_out).
    """
    m = A_out.shape[1]
    return (A_out - Y) / m


def _test_losses() -> None:
    y = np.array([0, 1, 2])
    Y = one_hot(y, n_classes=3)
    A_out = np.array([[0.7, 0.1, 0.2], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]).T

    print("one_hot:")
    print(Y)
    print("cross_entropy_loss:", cross_entropy_loss(A_out, Y))
    print("softmax_crossentropy_backward:")
    print(softmax_crossentropy_backward(A_out, Y))


if __name__ == "__main__":
    _test_losses()
