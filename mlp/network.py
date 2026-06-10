import numpy as np
import time

from mlp.activations import ACTIVATIONS, softmax
from mlp.losses import cross_entropy_loss, one_hot, softmax_crossentropy_backward
from mlp.optimizers import SGD, Optimizer


class MLP:
    """
    Multi-Layer Perceptron (MLP) implementado do zero com NumPy.
    Convenção de dimensões:
        Entrada X: (n_features, m) - exemplos em colunas.
        Parâmetros W[l]: (n_out, n_in)
        Bias b[l]: (n_out, 1)
    """

    def __init__(
        self,
        layer_sizes: list,
        activation: str = "relu",
        optimizer: Optimizer = None,
        seed: int = 42,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes deve ter ao menos 2 elementos (entrada e saída).")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Ativação '{activation}' desconhecida. Use: {list(ACTIVATIONS.keys())}")

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.activation_name = activation
        self.act_fn, self.act_backward = ACTIVATIONS[activation]
        self.optimizer = optimizer if optimizer is not None else SGD(learning_rate=0.01)

        # Histórico de métricas
        self.history = {
            "train_loss": [],
            "val_loss":   [],
            "train_acc":  [],
            "val_acc":    [],
        }

        # Inicializa parâmetros
        self._init_params(seed)

    def _init_params(self, seed: int) -> None:
        """
        Inicializa pesos com He normal (para ReLU) / Xavier (para a saída) e bias com zeros.
        """
        if seed is not None:
            np.random.seed(seed)

        self.W = []
        self.b = []

        for l in range(self.n_layers):
            n_in  = self.layer_sizes[l]
            n_out = self.layer_sizes[l + 1]

            # Inicialização He para as camadas ocultas, Xavier para a de saída
            if l == self.n_layers - 1:
                # Xavier
                limit = np.sqrt(1.0 / n_in)
            else:
                # He
                limit = np.sqrt(2.0 / n_in)

            W_l = np.random.randn(n_out, n_in) * limit
            b_l = np.zeros((n_out, 1))

            self.W.append(W_l)
            self.b.append(b_l)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Executa o forward pass e retorna a ativação final e o cache intermediário.
        """
        cache = {"Z": [], "A": [X]}  # A[0] = X (entrada)
        A_prev = X

        # Camadas ocultas
        for l in range(self.n_layers - 1):
            Z_l = self.W[l] @ A_prev + self.b[l]
            A_l = self.act_fn(Z_l)
            cache["Z"].append(Z_l)
            cache["A"].append(A_l)
            A_prev = A_l

        # Camada de saída (Softmax)
        l_out = self.n_layers - 1
        Z_out = self.W[l_out] @ A_prev + self.b[l_out]
        A_out = softmax(Z_out)
        cache["Z"].append(Z_out)
        cache["A"].append(A_out)

        return A_out, cache

    def backward(self, A_out: np.ndarray, Y: np.ndarray, cache: dict) -> tuple:
        """
        Executa o backward pass (backpropagation) e calcula os gradientes dW e db.
        """
        m = A_out.shape[1]
        dW = [None] * self.n_layers
        db = [None] * self.n_layers

        # Gradiente da camada de saída
        l_out = self.n_layers - 1
        dZ = softmax_crossentropy_backward(A_out, Y)  # dL/dZ_out: (n_classes, m)

        A_prev = cache["A"][l_out]
        dW[l_out] = dZ @ A_prev.T
        db[l_out] = np.sum(dZ, axis=1, keepdims=True)

        # Gradiente das camadas ocultas
        for l in range(l_out - 1, -1, -1):
            dA = self.W[l + 1].T @ dZ
            dZ = dA * self.act_backward(cache["Z"][l])  # Hadamard product
            A_prev = cache["A"][l]
            dW[l] = dZ @ A_prev.T
            db[l] = np.sum(dZ, axis=1, keepdims=True)

        return dW, db

    def _update_params(self, dW: list, db: list) -> None:
        """
        Intercala parâmetros e gradientes em uma lista linear para que o otimizador atualize.
        """
        params = []
        grads = []
        for l in range(self.n_layers):
            params.extend([self.W[l], self.b[l]])
            grads.extend([dW[l], db[l]])
        self.optimizer.update(params, grads)

    def save_weights(self, file_path: str) -> None:
        """
        Salva os pesos e bias em um arquivo NumPy .npz.
        """
        np.savez(file_path, W=self.W, b=self.b)

    def load_weights(self, file_path: str) -> None:
        """
        Carrega pesos e bias de um arquivo NumPy .npz.
        """
        data = np.load(file_path, allow_pickle=True)
        self.W = [arr for arr in data["W"]]
        self.b = [arr for arr in data["b"]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Gera as predições de classes para uma matriz de entrada X.
        """
        A_out, _ = self.forward(X)
        return np.argmax(A_out, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Avalia o custo e a acurácia sobre o conjunto de dados.
        """
        A_out, _ = self.forward(X)
        Y = one_hot(y, n_classes=self.layer_sizes[-1])
        loss = cross_entropy_loss(A_out, Y)
        preds = np.argmax(A_out, axis=0)
        accuracy = np.mean(preds == y)
        return loss, accuracy

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: bool = True,
    ) -> dict:
        """
        Treina a rede neural usando Mini-Batch SGD.
        """
        m = X_train.shape[1]

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Shuffle
            perm = np.random.permutation(m)
            X_shuffled = X_train[:, perm]
            y_shuffled = y_train[perm]

            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[start:end]
                Y_batch = one_hot(y_batch, n_classes=self.layer_sizes[-1])

                # Forward, backward, update
                A_out, cache = self.forward(X_batch)
                dW, db = self.backward(A_out, Y_batch, cache)
                self._update_params(dW, db)

            # Métricas da época
            train_loss, train_acc = self.evaluate(X_train, y_train)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            val_str = ""
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                val_str = f"  |  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"

            elapsed = time.time() - t0
            if verbose:
                print(
                    f"Epoca {epoch:3d}/{epochs}"
                    f"  train_loss: {train_loss:.4f}"
                    f"  train_acc: {train_acc:.4f}"
                    f"{val_str}"
                    f"  ({elapsed:.1f}s)"
                )

        return self.history

    def gradient_check(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1e-5,
        n_checks: int = 10,
    ) -> float:
        """
        Verifica a corretude do backpropagation comparando com o gradiente numérico (Diferenças Finitas).
        """
        Y = one_hot(y, n_classes=self.layer_sizes[-1])
        A_out, cache = self.forward(X)
        dW_analytic, db_analytic = self.backward(A_out, Y, cache)

        all_params = []
        all_grads = []
        for l in range(self.n_layers):
            all_params.append(self.W[l])
            all_grads.append(dW_analytic[l])
            all_params.append(self.b[l])
            all_grads.append(db_analytic[l])

        max_diff = 0.0
        np.random.seed(0)

        print(f"\n=== Gradient Check (epsilon={epsilon}, {n_checks} parametros) ===")

        for _ in range(n_checks):
            layer_idx = np.random.randint(0, len(all_params))
            param = all_params[layer_idx]
            grad = all_grads[layer_idx]
            idx = tuple(np.random.randint(0, s) for s in param.shape)

            original = param[idx]

            # f(x + epsilon)
            param[idx] = original + epsilon
            A_plus, _ = self.forward(X)
            loss_plus = cross_entropy_loss(A_plus, Y)

            # f(x - epsilon)
            param[idx] = original - epsilon
            A_minus, _ = self.forward(X)
            loss_minus = cross_entropy_loss(A_minus, Y)

            # Restaura
            param[idx] = original

            # Gradiente numérico
            grad_numeric = (loss_plus - loss_minus) / (2 * epsilon)
            grad_analytic = grad[idx]

            # Diferença relativa
            num = abs(grad_analytic - grad_numeric)
            den = abs(grad_analytic) + abs(grad_numeric) + 1e-15
            diff = num / den

            max_diff = max(max_diff, diff)
            status = "OK" if diff < 1e-4 else "ERRO"
            print(
                f"  param[{layer_idx}]{idx} | analitico: {grad_analytic:.6f}"
                f" | numerico: {grad_numeric:.6f}"
                f" | diff_rel: {diff:.2e}  {status}"
            )

        print(f"\nDiferenca maxima: {max_diff:.2e}")
        if max_diff < 1e-4:
            print("OK - Gradientes CORRETOS!\n")
        else:
            print("ERRO - Atencao: gradientes potencialmente incorretos!\n")

        return max_diff


def _test_network() -> None:
    from mlp.optimizers import SGD

    X = np.random.randn(4, 5)
    y = np.array([0, 1, 2, 1, 0])
    model = MLP([4, 8, 3], activation="relu", optimizer=SGD(learning_rate=0.01), seed=0)

    history = model.train(X, y, epochs=2, batch_size=2, verbose=False)
    loss, acc = model.evaluate(X, y)
    print(f"Treino rápido: loss={loss:.4f}, acc={acc:.4f}")
    print("Histórico:", history)


if __name__ == "__main__":
    _test_network()
