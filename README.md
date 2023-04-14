# Forest_cover_type_classification

1. Loaded the Covertype Data Set
    <https://archive.ics.uci.edu/ml/datasets/Covertype>

2. Implemented a very simple heuristic that will classify the data

    - Chose the most common cover type
3. Used Scikit-learn library to train two simple Machine Learning models

    - Chose logistic regression and decision tree as a baseline
4. Used TensorFlow library to train a neural network that classifies the data

    - Created a function that will find a good set of hyperparameters for the NN
    - Plotted training curves for the best hyperparameters
5. Evaluated neural network and other models
    - Used accuracy (optionally also precision, recall, and F1 score)

6. Created a very simple REST API that serve these models
    - Users can choose a model - heuristic, two other baseline models, or neural network
    - It takes all necessary input features and return a prediction
