import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP, SGDMomentum, load_mnist

plt.style.use('ggplot')
os.makedirs('results/assets', exist_ok=True)

X_train, y_train, X_test, y_test = load_mnist('./data')
val_size = 5000
X_val = X_train[:, -val_size:]
y_val = y_train[-val_size:]
X_tr = X_train[:, :-val_size]
y_tr = y_train[:-val_size]

configs = [
    ('Exp1', [784, 128, 64, 10], 0.01, 64, 20),
    ('Exp2', [784, 256, 128, 10], 0.01, 128, 20),
]

# Optional: gradient check numérico para validar o backprop antes do treino.
print('\n=== Gradient check numérico opcional ===')
grad_model = MLP([784, 128, 64, 10], activation='relu', optimizer=SGDMomentum(learning_rate=0.01, momentum=0.9), seed=0)
grad_model.gradient_check(X_tr[:, :10], y_tr[:10], epsilon=1e-5, n_checks=5)

rows = []
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
results = []
for name, layers, lr, batch, epochs in configs:
    print(f'Running {name}')
    model = MLP(layers, activation='relu', optimizer=SGDMomentum(learning_rate=lr, momentum=0.9), seed=0)
    history = model.train(X_tr, y_tr, epochs=epochs, batch_size=batch, X_val=X_val, y_val=y_val, verbose=False)
    val_loss, val_acc = model.evaluate(X_val, y_val)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    results.append({
        'name': name,
        'architecture': ' -> '.join(map(str, layers)),
        'lr': lr,
        'batch_size': batch,
        'epochs': epochs,
        'history': history,
        'test_loss': test_loss,
        'test_acc': float(test_acc),
        'val_acc': float(val_acc),
        'val_loss': float(val_loss),
        'model': model,
    })
    rows.append([name, '->'.join(map(str, layers)), lr, batch, epochs, test_acc, test_loss, val_acc, val_loss])
    colors = ['#1f77b4', '#ff7f0e']
    val_colors = ['#17becf', '#d62728']
    ix = len(results) - 1
    ax[0].plot(history['train_loss'], label=f'{name} train', color=colors[ix])
    ax[1].plot(history['train_acc'], label=f'{name} train', color=colors[ix])
    ax[1].plot(history['val_acc'], '--', label=f'{name} val', color=val_colors[ix])
    print(f'{name} test_acc={test_acc:.4f} test_loss={test_loss:.4f} val_acc={val_acc:.4f}')

ax[0].set_title('Training Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
fig.tight_layout()
fig.savefig('results/assets/curvas_treino.png')
plt.close(fig)

with open('results/comparacao_experimentos.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Experiment', 'Architecture', 'Learning Rate', 'Batch Size', 'Epochs', 'Test Accuracy', 'Test Loss', 'Val Accuracy', 'Val Loss'])
    writer.writerows(rows)

best = max(results, key=lambda r: r['test_acc'])
print(f"Best experiment: {best['name']} with test_acc={best['test_acc']:.4f}")

# Confusion matrix with NumPy
cm = np.zeros((10, 10), dtype=int)
Y_pred = best['model'].predict(X_test)
for t, p in zip(y_test, Y_pred):
    cm[int(t), int(p)] += 1

# Optional: PCA das ativações internas da última camada oculta

def plot_hidden_activations_pca(model, X_data, y_data, file_path, n_samples=1000):
    indices = np.random.choice(X_data.shape[1], size=min(n_samples, X_data.shape[1]), replace=False)
    X_sample = X_data[:, indices]
    y_sample = y_data[indices]
    _, cache = model.forward(X_sample)
    A_hidden = cache['A'][-2]
    A_centered = A_hidden - np.mean(A_hidden, axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
    projected = (S[:2, None] * Vt[:2, :]).T

    fig, ax = plt.subplots(figsize=(10, 8))
    for digit in np.unique(y_sample):
        mask = (y_sample == digit)
        ax.scatter(projected[mask, 0], projected[mask, 1], s=12, alpha=0.75, label=str(digit))
    ax.legend(title='Dígito', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('PCA das ativações da última camada oculta')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)

plot_hidden_activations_pca(best['model'], X_test, y_test, 'results/assets/activations_pca.png')

# Optional: summary of the most common confusion pairs
errors = []
for i in range(10):
    for j in range(10):
        if i != j:
            errors.append((cm[i, j], i, j))
errors.sort(reverse=True)
print('\nTop 5 pares mais confundidos:')
for count, actual, pred in errors[:5]:
    print(f'  {actual} → {pred}: {count} exemplos')

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title('Matriz de Confusão')
fig.colorbar(im, ax=ax)
ax.set_xlabel('Classe prevista')
ax.set_ylabel('Classe verdadeira')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()
fig.savefig('results/assets/confusion_matrix.png')
plt.close(fig)

errors = np.where(Y_pred != y_test)[0]
selected = errors[:9]
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, idx in zip(axes.flatten(), selected):
    img = X_test[:, idx].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'true={y_test[idx]} pred={Y_pred[idx]}')
    ax.axis('off')
plt.tight_layout()
fig.savefig('results/assets/exemplos_erro.png')
plt.close(fig)

print('Saved results/comparacao_experimentos.csv, results/assets/curvas_treino.png, results/assets/confusion_matrix.png, results/assets/exemplos_erro.png, results/assets/activations_pca.png')
