### Basics

Implementaton of basic perceptron from scratch, as well as multilayer perceptron (using Keras) to test how neural networks can handle different types of image data.

#### Files:
- `iris_perceptron.py` - basic implementation of perceptron using only numpy and basic math.
- `mnist_mlp.py` - neural network code for MNIST digits dataset.
- `out/` - folder that is automatically generated and it contains our results.
- `example-out/` - results from running this on my machine.

### Perceptron
I use Rosenblatt perception formula `y = w * x + b`, and I also added learning decay to prevent model from bouncing around. This perceptron is specifically used for distinguishing between setosa and non-setosa flowers.

#### Results:
Model works perfectly, it converged quickly and reached 100% accuracy on the test set.

### Multiplayer Perceptron
For this, I used Keras to build a neural network to classify handwritten digits. I also implemented batch normalisation and dropout to improve our results.

#### Expriments:

**Default:** I trained model on regular 28x28px images of digits, achieving 98% accuracy.

**Cut out:** In second task we crop out 10px from every side of image, resulting in 8x8 pixel image, significantly reducing data. The accuracy dropped to 86%, but it is still quite resilient, as most of digit is in the middle of it.

**Scrambled (Permuted):** I shuffled pixels in each image in exact same random order. While to normal person it looks like noise, model is still able to distinguish numbers. It also prooves that our network is not perceiving numbers as shapes, but rather patterns of independent pixel values.

### Running:
To run:
```sh
# Create virtual environment and install dependencies
python -m venv .venv
pip install -r requirements.txt

# Running perceptron
python iris_perceptron.py

# Running neural network
python mnist_mlp.py
```
You can check `out/` afterwards to see logs and plots.

<small>Project by Antoniuk Orest</small>\
<small>for Introducton to Machine Learning, UEK</small>