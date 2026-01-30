import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# configuration and creation of initial files
OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "mnist_results_summary.txt")

def log(message):
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

# helpers
class ImageTransformer:

    @staticmethod
    def crop_center(images, crop_pixels=10):
        h, w = images.shape[1], images.shape[2]
        return images[:, crop_pixels:h-crop_pixels, crop_pixels:w-crop_pixels]

    @staticmethod
    def permute_pixels(images, seed=42):
        rng = np.random.RandomState(seed)
        n, h, w = images.shape
        flat_size = h * w

        permutation = rng.permutation(flat_size)
        
        images_flat = images.reshape(n, -1)
        images_permuted = images_flat[:, permutation]
        return images_permuted.reshape(n, h, w)
        
# our magic
def build_modern_mlp(input_shape):
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        
        layers.Dense(256, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.Dropout(0.3),
        
        layers.Dense(128, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.Dropout(0.3),
        
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run_experiment(X_train, y_train, X_test, y_test, name):
    log(f"\n--- Running Experiment: {name} ---")
    log(f"Input Shape: {X_train.shape[1:]}")
    
    model = build_modern_mlp(X_train.shape[1:])

    stopper = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=4, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=128,
        validation_split=0.1,
        callbacks=[stopper],
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    log(f"Test Accuracy: {acc*100:.2f}%")

    # saving learning curve
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Learning Curve: {name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"learning_curve_{name.lower()}.png"))
    plt.close()

    return acc

def main():
    (X_train_full, y_train), (X_test_full, y_test) = keras.datasets.mnist.load_data()

    # normalisation
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test_full = X_test_full.astype('float32') / 255.0

    results = {}

    # experiment 1: baseline
    results['Baseline'] = run_experiment(X_train_full, y_train, X_test_full, y_test, "Baseline")

    # experiment 2: center 8x8
    X_train_crop = ImageTransformer.crop_center(X_train_full, crop_pixels=10)
    X_test_crop = ImageTransformer.crop_center(X_test_full, crop_pixels=10)
    results['Cropped'] = run_experiment(X_train_crop, y_train, X_test_crop, y_test, "Cropped")

    # experiment 3: permuted (scramble)
    X_train_perm = ImageTransformer.permute_pixels(X_train_full)
    X_test_perm = ImageTransformer.permute_pixels(X_test_full)
    results['Permuted'] = run_experiment(X_train_perm, y_train, X_test_perm, y_test, "Permuted")

    # final comparison
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    values = [v * 100 for v in results.values()]

    bars = plt.bar(names, values, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.ylabel('Accuracy (%)')
    plt.title('Impact of Transformations on MLP Performance')
    plt.ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom')
                 
    plt.savefig(os.path.join(OUTPUT_DIR, "final_comparison.png"))
    print(f"\nAll experiments done. Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
