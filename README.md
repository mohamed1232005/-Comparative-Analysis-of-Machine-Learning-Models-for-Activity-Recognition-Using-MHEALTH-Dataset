# -Comparative Analysis of Machine Learning Models for Activity Recognition Using MHEALTH Dataset
**Project Overview** :
This project implements machine learning models to perform activity recognition using the MHEALTH dataset. The objective was to compare multiple machine learning models—K-Nearest Neighbors (KNN), Linear Regression, Support Vector Machines (SVM), Neural Networks, and Logistic Regression—to determine which model performs best .

# Dataset: MHEALTH Dataset:
The MHEALTH dataset was chosen due to its relevance in mobile health applications. It contains:

Size: The dataset includes sensor readings captured during different physical activities.
Sensors: Accelerometers, gyroscopes, and magnetometers were used to collect motion and orientation data.
Data: Each entry in the dataset includes a timestamped recording of sensor data corresponding to specific physical activities, making it suitable for activity recognition tasks.


# Libraries and Tools Used
The following Python libraries were used to build, train, and evaluate the machine learning models:

**Pandas**: For data manipulation and preprocessing.
**NumPy**: For numerical computations.
**Scikit-learn**: For model implementation, training, and evaluation.
**Matplotlib & Seaborn**: For data visualization and understanding model performance.
**PyTorch (for Neural Networks)**: Used for building and training the neural network model.



# Project Workflow:


1.**Data Preprocessing**:
 The raw MHEALTH dataset required significant preprocessing to standardize and clean the data before applying machine learning models:

Dropping irrelevant columns: The 'subject' column was removed as it did not contribute to activity recognition.
Sampling the dataset: Due to the large size of the dataset, a representative subset was taken to reduce computational overhead without compromising model accuracy.
Data normalization: To ensure consistent scaling across features, data normalization was applied to the sensor readings.

2-**Exploratory Data Analysis (EDA)**: Exploratory analysis was conducted to understand the dataset and detect any irregularities. The steps included:

Heatmap visualization: A correlation matrix was generated to identify relationships between numerical columns, helping in feature selection.
Histograms: Generated to visualize the distribution of each feature, providing insights into sensor data patterns during different activities.

# 3-**Model Implementation** :
Each machine learning model was trained and evaluated to determine its effectiveness in classifying physical activities. Below are the models used and a description of their strengths and weaknesses:

-**K-Nearest Neighbors (KNN):**

Strengths: KNN is simple to implement and works well with smaller datasets. It does not require assumptions about data distribution.
Weaknesses: KNN suffers from scalability issues. It is computationally expensive for large datasets and is sensitive to noise and outliers.
Hyperparameter Tuning: The number of neighbors (k) was fine-tuned to optimize model performance.


-**Support Vector Machine (SVM):**

Strengths: SVM excels in high-dimensional spaces and can define complex decision boundaries using kernel functions.
Weaknesses: Requires careful tuning of hyperparameters (C and kernel choice). It is computationally expensive for large datasets.
Kernel Trick: Used various kernels (e.g., linear, radial basis function) to handle non-linear patterns in the sensor data.



-**Neural Networks:**

Strengths: Neural networks are highly effective for modeling non-linear relationships, especially in complex datasets like MHEALTH.
Weaknesses: Requires extensive data and computational resources to prevent overfitting and optimize performance.
Architecture: A multi-layer neural network was implemented using PyTorch, with tuning of hidden layers and neuron counts to enhance model accuracy.



-**Linear Regression:**

Strengths: Linear regression is computationally efficient and interpretable, making it a good baseline model for comparison.
Weaknesses: Assumes a linear relationship between features and the outcome, which may not accurately represent complex data like sensor readings.
Evaluation: The model was evaluated using Mean Squared Error (MSE) to measure prediction error.


-**Logistic Regression:**

Strengths: Provides probabilistic predictions, making it useful for activity recognition with an output of class probabilities. It is efficient and simple to implement.
Weaknesses: Assumes a linear decision boundary, which limits its performance on complex datasets.
Evaluation: Evaluated using classification metrics like accuracy, precision, and recall.

# 4-Model Evaluation :
Each model was evaluated based on the following metrics to determine performance:

-Accuracy: Measures the proportion of correctly classified activities.
-Precision: Measures the ratio of true positive predictions to the total number of predicted positives.
-Recall: Measures the model’s ability to identify all relevant instances (true positives).
-F1-score: A harmonic mean of precision and recall, giving a balanced measure of the model's performance.
-Confusion Matrix: Used for classification models to understand the distribution of true positive, false positive, true negative, and false negative predictions.
-Mean Squared Error (MSE): Applied to the Linear Regression model to assess prediction error.



# 5-**Results and Analysis**:

-The Neural Network model performed the best, achieving the highest accuracy and balanced precision-recall across activities.
-SVM performed well but required extensive tuning to avoid overfitting in high-dimensional space.
-KNN was limited by its computational inefficiency and sensitivity to noisy data.
-Linear Regression and Logistic Regression underperformed due to their linear assumptions, making them less suitable for this complex dataset.




# 6-**User Interface**:
 While no web interface was developed for this project, visualizations of model performance and confusion matrices were generated using Matplotlib and Seaborn, providing insights into model effectiveness and error distribution.




# **Techniques and Models**:
*TF-IDF for Feature Representation: Although not directly applied in this project, TF-IDF is commonly used in text-based datasets to represent term importance. For future projects involving text data, this method can be explored for feature representation.

*Cross-Validation and Hyperparameter Tuning: Cross-validation was used to ensure that model performance generalized well to unseen data. Grid search was applied to optimize hyperparameters like the number of neighbors in KNN and the kernel type for SVM.

*Neural Networks with PyTorch: PyTorch was used to build a deep neural network for activity classification. The model architecture included multiple hidden layers, and regularization techniques like dropout were applied to prevent overfitting.






The project demonstrated that neural networks, when tuned properly, outperform traditional machine learning models for complex tasks like activity recognition. The comparison of models provided valuable insights into the trade-offs between model complexity, computational efficiency, and accuracy.

For future work, additional methods such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) could be explored to further improve activity recognition accuracy, especially for sequential data. Additionally, real-time prediction could be implemented for applications in mobile health monitoring.
