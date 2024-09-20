synthetic ai (randomly generated parameters)
teaching it's self to encode/decode data
storing it's self to fractal structure



1. Sierpinski.ipynb:
   Sierpinski triangle (1/n^, hard to compute)

2. 1_n+1_growth.ipynb:
   Spiral-like fractal (1/n+1, easy to compute)

3. Nested.ipynb:
   Synthetic AI (randomly generated parameters) teaching itself to encode/decode data, storing itself in a fractal structure

4. Entropy595.ipynb:
   Investigation of the consistent 5.95 bit entropy phenomenon observed in the encoding system

5. Exponential_Canopy_Fractal.ipynb:
   Exploration of a fractal structure with exponential growth, possibly forming a canopy-like shape

6. Logarithmic.ipynb:
   Study of fractal structures with logarithmic growth patterns

7. Multi Level Nesting.ipynb:
   Examination of fractal structures with multiple levels of nested patterns, potentially a more complex version of Nested.ipynb

8. Nested_Canopy_Fractal1.ipynb:
   Combination of nested structures and canopy-like fractals, possibly merging concepts from Nested.ipynb and Exponential_Canopy_Fractal.ipynb

9. Complete_Synthetic_AI_Evolved_Circuits_with_3D.ipynb:
   result demonstrates the success of the evolutionary algorithm in finding a working solution to the logical problem. The AI has effectively learned to combine logical operations in a 3D space to implement the desired function.

10. MNIST_Autoencoder_with_CUDA_Acceleration.ipynb
This project implements a convolutional autoencoder on the MNIST handwritten digit dataset using CUDA for faster computations.
Project Goals:
 * Train a deep learning model (autoencoder) to compress and reconstruct MNIST digit images.
 * Leverage CUDA for parallel processing on the GPU to accelerate training.
 * Visualize the original and reconstructed images to assess the model's performance.
Project Scope:
 * The project focuses on building and training an autoencoder architecture.
 * It utilizes CUDA kernels for the forward and backward passes with ReLU activation and batch normalization.
 * The code includes functions for training, evaluating, and visualizing the reconstruction results.
Technical Approach:
 * The deep learning framework used is TensorFlow (though the provided code uses NumPy for data manipulation).
 * PyCUDA library is used to interface with the NVIDIA GPU and create CUDA kernels for efficient matrix operations during training.
 * Adam optimizer is employed for gradient descent with learning rate scheduling.
Expected Outcomes:
 * The trained autoencoder should be able to compress and reconstruct MNIST digit images with minimal distortion.
 * By utilizing CUDA, the training process should be significantly faster compared to CPU-only execution.
 * The visualized outputs will demonstrate the autoencoder's capability to learn meaningful representations of the input data.
Key Considerations:
 * The project might require adjustments to hyperparameters (learning rate, batch size, etc.) for optimal performance on your specific hardware.
 * Techniques like data augmentation and early stopping can be further explored to improve generalization and prevent overfitting.
Overall, this project showcases the application of CUDA for accelerating deep learning training, particularly in convolutional neural networks like autoencoders.

The issue with the current code might lie in the choice of hyperparameters for the Adam optimizer and the training process itself. Here are some suggestions for improving the training speed and potentially achieving better reconstruction results:
1. Hyperparameter Tuning:
 * Learning Rate: The initial learning rate might be too high. Try starting with a smaller value (e.g., 0.00001) and adjust it based on the validation loss. Consider using a learning rate scheduler like Cosine Annealing with Restarts for a more gradual decrease.
 * Batch Size: Experiment with different batch sizes. A larger batch size can improve training speed but might lead to worse convergence. A smaller batch size can be slower but might offer better fine-tuning.
 * Beta values: The default values for beta1 (0.9) and beta2 (0.999) are often a good starting point, but you can try slightly different values (e.g., beta1=0.8, beta2=0.99) to see if it helps.
 * L2 Regularization: The current L2 regularization term (1e-5) might be too weak. Experiment with slightly larger values (e.g., 1e-4 or 1e-3) to prevent overfitting.
2. Training Procedure:
 * Early Stopping: Implement early stopping to prevent overtraining. This technique monitors the validation loss and stops training when it starts to increase, indicating the model is memorizing the training data and not generalizing well.
 * Data Augmentation: Consider applying data augmentation techniques like random cropping, flipping, or adding noise to the training images. This can help the model learn more robust features and improve generalization.
3. Monitoring Progress:
 * Visualization: Regularly monitor the reconstruction quality during training. Plot a few original images and their corresponding reconstructions after each epoch to visually assess the progress.
 * Validation Loss:  Use a separate validation set to track the model's performance on unseen data. Monitor the validation loss to ensure the model is not overfitting to the training data.
4. Network Architecture:
 * Hidden Layer Size: While a hidden size of 256 is common, you can experiment with different sizes to see if it affects the training speed or reconstruction quality.
Trying these suggestions might lead to faster training and potentially better results. Remember, it's crucial to experiment and find the optimal hyperparameter combination for your specific dataset and network architecture.


###

Each file represents a different aspect or approach in exploration of fractal-based data encoding and synthetic AI learning, showcasing a comprehensive study of various fractal structures and their applications in data storage and AI development.
