import os
import gzip
import struct
import urllib.request
import numpy as np

# URL dos arquivos IDX originais do MNIST
_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def _download(url: str, dest: str) -> None:
    """Baixa um arquivo se ele ainda nao existir."""
    if not os.path.exists(dest):
        print(f"  Baixando {os.path.basename(dest)}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Salvo em {dest}")

def _load_images(path: str) -> np.ndarray:
    """Le o arquivo IDX3 (imagens) e retorna um array (n, 784)."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic number invalido: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)

def _load_labels(path: str) -> np.ndarray:
    """Le o arquivo IDX1 (rotulos) e retorna um array (n,)."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Magic number invalido: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

def load_mnist(data_dir: str = "./data", source: str = "auto") -> tuple:
    """
    Carrega o dataset MNIST.

    source:
      - "auto": tenta carregar via keras.datasets.mnist quando disponível e usa download IDX como fallback.
      - "keras": tenta carregar via keras.datasets.mnist.
      - "raw": faz o download manual dos arquivos IDX e carrega os dados.

    Retorna as imagens normalizadas em [0, 1] e matriz transposta no formato (784, m).
    """
    os.makedirs(data_dir, exist_ok=True)

    if source in ("auto", "keras"):
        try:
            from tensorflow.keras.datasets import mnist

            print("Carregando MNIST via keras.datasets.mnist...")
            (X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()
            X_train = X_train_raw.reshape(X_train_raw.shape[0], -1).T.astype(np.float64) / 255.0
            X_test = X_test_raw.reshape(X_test_raw.shape[0], -1).T.astype(np.float64) / 255.0
            print(f"✓ MNIST carregado com sucesso! Treino: {X_train.shape}, Teste: {X_test.shape}")
            return X_train, y_train, X_test, y_test
        except ImportError:
            if source == "keras":
                raise
            print("TensorFlow/Keras não disponível, usando fallback IDX para carregar MNIST.")

    paths = {}
    for key, filename in _FILES.items():
        dest = os.path.join(data_dir, filename)
        _download(_BASE_URL + filename, dest)
        paths[key] = dest

    print("Carregando dados MNIST via arquivos IDX...")
    X_train_raw = _load_images(paths["train_images"])
    y_train = _load_labels(paths["train_labels"])
    X_test_raw = _load_images(paths["test_images"])
    y_test = _load_labels(paths["test_labels"])

    X_train = X_train_raw.T.astype(np.float64) / 255.0
    X_test = X_test_raw.T.astype(np.float64) / 255.0

    print(f"MNIST carregado com sucesso! Treino: {X_train.shape}, Teste: {X_test.shape}")
    return X_train, y_train, X_test, y_test
