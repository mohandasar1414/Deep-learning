import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Step 2: Preprocess the data
# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first image in the training set for visual check
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.title(f"Label: {train_labels[0]}")
plt.show()

# Step 3: Build the deep learning model
model = models.Sequential([
    # Flatten the 28x28 image into a 1D vector
    layers.Flatten(input_shape=(28, 28)),
    
    # Fully connected layer with 128 neurons and ReLU activation function
    layers.Dense(128, activation='relu'),
    
    # Output layer with 10 neurons for the 10 classes
    layers.Dense(10, activation='softmax')  # Softmax is used for multi-class classification
])

# Step 4: Compile the model
model.compile(optimizer='adam',  # Optimizer to update the weights
              loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
              metrics=['accuracy'])  # Metric to evaluate the model

# Step 5: Train the model
model.fit(train_images, train_labels, epochs=5)  # Training the model for 5 epochs

# Step 6: Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Step 7: Make predictions
predictions = model.predict(test_images)

# Display the first prediction along with the corresponding image
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.title(f"Predicted label: {np.argmax(predictions[0])}, True label: {test_labels[0]}")
plt.show()
