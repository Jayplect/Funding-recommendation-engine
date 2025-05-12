# Funding Recommendation Engine (Machine Learning, Neural Networks, TensorFlow)

## Overview of the Project
The nonprofit foundation **Alphabet Soup** wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. For this project, I built a **binary classifier** to predict the success of applicants seeking funding from Alphabet Soup. Leveraging the features in the dataset, the model uses **machine learning** and **neural networks** to make accurate predictions.

## Dependencies used
![TENSORFLOW](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![SkLearn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)

## Data Description
The dataset, a CSV, contained more than **34,000 organizations** that have received funding from Alphabet Soup over the years. Within this dataset are a number of features that capture metadata about each organization, such as:

- **EIN and NAME**—Identification columns
- **APPLICATION_TYPE**—Alphabet Soup application type
- **AFFILIATION**—Affiliated sector of industry
- **CLASSIFICATION**—Government organization classification
- **USE_CASE**—Use case for funding
- **ORGANIZATION**—Organization type
- **STATUS**—Active status
- **INCOME_AMT**—Income classification
- **SPECIAL_CONSIDERATIONS**—Special considerations for application
- **ASK_AMT**—Funding amount requested
- **IS_SUCCESSFUL**—Was the money used effectively (target variable)

## Results

### Data Preprocessing
In the data preprocessing stage, I prepared the dataset for training and testing the classifier. The target variable was **IS_SUCCESSFUL**, while the rest of the dataset formed the features. I removed identification features and performed the following steps:

1. **Handled Missing Values**: Checked for and filled missing values where necessary.
2. **Encoded Categorical Variables**: Used **one-hot encoding** for categorical features like `APPLICATION_TYPE`, `CLASSIFICATION`, and `AFFILIATION`.
3. **Scaled Numerical Features**: Standardized the numerical values (e.g., `ASK_AMT`) to ensure consistency in model training.
4. **Binned Features**: For features where exact values weren't as important, I binned categories (e.g., `APPLICATION_TYPE` and `CLASSIFICATION`) to reduce noise in the dataset.
   
   Example of binning:
   ```python
   # Check the value counts for binning
   unique_fields = df['field'].value_counts()

   # Set a cutoff value and create a list of fields to replace
   cutoff_value = 10  # Frequency threshold for binning
   field_to_replace = list(unique_fields[unique_fields < cutoff_value].index)

   # Replace in the dataframe
   for field in field_to_replace:
       df['FIELD'] = df['FIELD'].replace(field, "Other")

   # Verify that binning was successful
   df['FIELD'].value_counts()

- Compiling, Training, and Evaluating the Model  
I used TensorFlow and Keras to build the neural network. The model was defined using the Sequential API, with the following architecture:

Input layer -> 2 hidden layers with ReLU activation (80 and 30 neurons)

Output layer with 1 node and sigmoid activation for binary classification

Structure of model

  <img width="500" alt="image" src="https://github.com/Jayplect/deep-learning-AlphabetSoup/assets/107348074/2c3ad445-711f-4318-ba83-f1d77068d0c7">

Training: The model was trained for 55 epochs, with a batch size of 32, and achieved the following performance:

Loss: 0.566

Accuracy: 72.6%

Model Checkpointing: To ensure that the model’s weights were saved periodically, I used a ModelCheckpoint callback to save weights every 5 epochs:
checkpoint_path = "./training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Calculate the batch size and steps per epoch
batch_size = 32
n_batches = len(X_train_scaled)/batch_size
save_period = 5

# Callback to save model weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    steps_per_epoch= 5,
    verbose=1, 
    save_weights_only=True,
    save_freq= int(save_period*n_batches))

## Optimization
I made several adjustments to the model to optimize performance:

- **Included Additional Features**: I added the **Name** and **Affiliation** features, binning them into the "Other" category as explained earlier.
- **Increased Hidden Layer Size**: The first hidden layer was increased to 100 neurons, the second to 40 neurons, and a third hidden layer was added.
- **Increased Epochs**: The training was extended to 100 epochs for further optimization.

Structure of Optimized model

<img width="500" alt="image" src="https://github.com/Jayplect/deep-learning-AlphabetSoup/assets/107348074/993d43c8-76d7-4e9a-8b5f-038865c6fb99">

After optimization, the model significantly improved:

- **Loss**: 0.4723
- **Accuracy**: 77.22%

### Model Structure (Optimized)
**Optimized Architecture**:

- Input layer -> Hidden layer 1 (100 neurons, ReLU) -> Hidden layer 2 (40 neurons, Tanh) -> Hidden layer 3 (40 neurons, ReLU)
- Output layer (1 node, Sigmoid)

## Final Model Evaluation
The deep learning model outperformed the baseline, improving accuracy by ~5% and reducing loss by ~10%.

## Real-world Impact
The model helps Alphabet Soup allocate funding more effectively by identifying the most promising applicants based on historical data.

## Future Directions
To further optimize this model, I recommend exploring ensemble methods like Random Forests for comparison with the neural network's performance.

## Conclusion
This project showcases the power of deep learning and machine learning to address real-world problems, providing a data-driven approach to funding decisions that will drive social impact.

## References
- IRS. Tax Exempt Organization Search Bulk Data Downloads.
