import numpy as np

def relu(z):
    """
    Calcula a ativação ReLU (Rectified Linear Unit) de forma elemento a elemento.

    Fórmula:
        f(z) = max(0, z)

    Parâmetros:
        z (np.ndarray): Tensor de entrada (pode ser vetor ou matriz).

    Retorna:
        np.ndarray: Ativações resultantes com o mesmo formato de z.
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Calcula a derivada da função de ativação ReLU.

    Fórmula:
        f'(z) = 1 se z > 0
        f'(z) = 0 se z <= 0

    Parâmetros:
        z (np.ndarray): Entrada (pré-ativação) que foi fornecida à função ReLU.

    Retorna:
        np.ndarray: Gradientes elemento a elemento com o mesmo formato de z.
    """
    return np.where(z > 0, 1.0, 0.0)

def softmax(z):
    """
    Calcula a função de ativação Softmax para um vetor ou matriz de logits.
    Implementada de forma numericamente estável para evitar overflow.

    Fórmula estável:
        f(z_i) = exp(z_i - max(z)) / soma_j(exp(z_j - max(z)))

    Parâmetros:
        z (np.ndarray): Logits de entrada. 
                        Pode ser 1D (num_classes,) ou 2D (batch_size, num_classes).

    Retorna:
        np.ndarray: Distribuição de probabilidade resultante com o mesmo formato de z.
    """
    if z.ndim == 1:
        # Estabilidade numérica: subtrai o máximo de z
        z_shifted = z - np.max(z)
        exps = np.exp(z_shifted)
        return exps / np.sum(exps)
    else:
        # Para matrizes (lote de exemplos): subtrai o máximo ao longo do eixo das classes (axis=1)
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(z_shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)
