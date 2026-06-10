"""Ponto de entrada do pacote mlp."""

from . import MLP, load_mnist, SGD, SGDMomentum


def main() -> None:
    print("Pacote mlp do zero")
    print("Importe MLP, SGD, SGDMomentum e load_mnist para treinar modelos:")
    print("  from mlp import MLP, SGD, SGDMomentum, load_mnist")
    print("Use os módulos individuais para testar implementações:")
    print("  python -m mlp.activations")
    print("  python -m mlp.losses")
    print("  python -m mlp.optimizers")
    print("  python -m mlp.network")


if __name__ == "__main__":
    main()
