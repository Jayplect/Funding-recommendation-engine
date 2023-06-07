## Overview of the Analysis
The nonprofit foundation Alphabet Soup (a fictional company) wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With my knowledge of machine learning and neural networks, I utilized the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

## Dependencies used

![TENSORFLOW](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![SkLearn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)

## Data Description
From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
`EIN and NAME`—Identification columns,
`APPLICATION_TYPE`—Alphabet Soup application type,
`AFFILIATION`—Affiliated sector of industry,
`CLASSIFICATION`—Government organization classification,
`USE_CASE`—Use case for funding,
`ORGANIZATION`—Organization type,
`STATUS`—Active status,
`INCOME_AMT`—Income classification,
`SPECIAL_CONSIDERATIONS`—Special considerations for application,
`ASK_AMT`—Funding amount requested,
`IS_SUCCESSFUL`—Was the money used effectively.

## Results

- Data Preprocessing

Under this step, I prepared the dataset that will be used for training and testing the classifier. Because of the question I wanted to answer, I selected `IS_SUCCESSFUL` as the target of my model while the remaining dataset formed the features. After inspecting if the dataset contained features relevant to predicting the success of applicants, I decided to remove the identification features. Preprocessing steps also included checking for and handling missing values, encoding categorical variables, and scaling numerical features. In addition, I observed that the exact value of some of the features was not as important as the general range or category they fall into. Accordingly, in order to smoothen out noise in the dataset I binned some of the features (`APPLICATION_TYPE` and `CLASSIFICATION`) by grouping similar values in feature together.
    
   - example of binning  
    
          # look at field value counts for binning
          unique_fields = df.field.value_counts()

          # Choose a cutoff value and create a list of classifications to be replaced
          cutoff_value = value
          field_to_replace = list(unique_fields[unique_fields < cutoff_value].index)

          # Replace in dataframe
          for field in field_to_replace:
              df['FIELD'] = df['FIELD'].replace(field,"Other")

          # Check to make sure binning was successful
          df['FIELD'].value_counts()

- Compiling, Training, and Evaluating the Model
To build the neural network, I imported the necessary Tensorflow and Keras modules. I then defined the neural network architecture using the Sequential model from keras. I added two layers with similar activation function (i.e., `relu`) using the `Dense` class. The number of neurons were set to 80 and 30 for the first and second hidden layer, respectively (Fig 1).

Fig 1: Structure of model

  <img width="500" alt="image" src="https://github.com/Jayplect/deep-learning-AlphabetSoup/assets/107348074/2c3ad445-711f-4318-ba83-f1d77068d0c7">

Lastly, I created a call to save the model's weight during execution for every five epochs. This was to ensure that the model could be restored from it's weight.
   
  - example of call back
    
        # Include the epoch in the file name
        checkpoint_path = "./training_2/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Calculate the batch size to save per epoch
        batch_size = 32
        n_batches = len(X_train_scaled)/batch_size
        save_period = 5

        # Create a callback that saves the model's weights every 10 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            steps_per_epoch= 5,
            verbose=1, 
            save_weights_only=True,
            save_freq= int(save_period*n_batches))

After 55 epochs, the loss and accuracy for the model was 0.566 and 0.726, respectively.

  - Snapshot of model perfomance
  
<img width="500" alt="image" src="https://github.com/Jayplect/deep-learning-AlphabetSoup/assets/107348074/d8d5c216-4cda-45b0-967a-182be68ede8e">

- Optimization 

Fig 2: Structure of Optimized model

In order to optimize the model's performance, I increased the model hyperparameters 

Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.







## References
Data for this dataset was generated by edX Boot Camps LLC, and is intended for educational purposes only.
