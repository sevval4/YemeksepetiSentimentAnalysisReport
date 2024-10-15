# Yemeksepeti Sentiment Analysis Project

## Introduction
Yemeksepeti is a platform where customers can leave reviews about the restaurants they order from, providing valuable feedback on aspects like food quality, delivery speed, and service. These reviews are ideal for sentiment analysis as they reflect customer satisfaction or dissatisfaction.

The main goal of this project is to analyze customer reviews on Yemeksepeti to understand customer perceptions. We cleaned and preprocessed the data and used machine learning techniques to extract insights from these reviews. The primary algorithm used was the Naive Bayes classifier, optimized to predict sentiment classes in the reviews.

## Dataset
The dataset consists of customer reviews from Yemeksepeti, including the following columns:
- **Speed:** User ratings of delivery speed.
- **Service:** User ratings of service quality (e.g., packaging, delivery behavior).
- **Flavour:** User ratings of food taste and quality.
- **Review:** Written comments on the food delivery experience.

The dataset has a total of 51,922 rows. Since some reviews may be empty or contain irrelevant information, data cleaning and preprocessing were essential before further analysis.

## Data Cleaning and Preprocessing
The data was processed using the following steps:
- **Handling Missing Reviews:** Empty reviews were filled with "No comment available."
- **Text Preprocessing:**
  - Converted text to lowercase.
  - Removed Turkish stop words (e.g., "ve," "ile," "bu").
  - Stemmed words to their root forms (e.g., "yemekler" → "yemek").

## Model Selection
Machine learning classification models were used to perform sentiment analysis. The primary model used was the Naive Bayes classifier due to its simplicity and effectiveness with text-based data.

### Naive Bayes Classifier
Multinomial Naive Bayes was chosen for its performance on text classification tasks. Reviews were converted into numerical vectors using word frequencies, which allowed the model to predict sentiment classes accurately.

## Model Training and Evaluation
The data was split into 70% training and 30% testing. We used the TF-IDF (Term Frequency-Inverse Document Frequency) method to convert reviews into numerical vectors. The model was trained on the training set and evaluated using metrics like accuracy, precision, F1 score, and a confusion matrix.

### Model Performance
The Naive Bayes model demonstrated high accuracy:
- **Training Data Accuracy:** 82.5%
- **Test Data Accuracy:** 87.2%

The close accuracy rates between training and testing indicate that the model avoided overfitting and generalized well.

### Cross-Validation Results
The cross-validation results were consistent:
- **Cross-Validation Score (Training Data):** 89.1%
- **Cross-Validation Score (Test Data):** 88.8%

### Strengths and Weaknesses
- **Strengths:**
  - Fast and effective for text data.
  - TF-IDF improved feature extraction by highlighting important words.
- **Weaknesses:**
  - Struggles with complex phrases or figurative language.
  - Assumes words are independent of each other, which may lead to incorrect predictions.

## Experimental Evaluation
From the experiments, we observed that the Naive Bayes classifier is effective for sentiment analysis of Yemeksepeti reviews. The model can quickly analyze customer feedback, providing potential insights for restaurants and users.

## TF-IDF Vectorization
The TF-IDF method was used to determine the importance of words in the reviews, considering both their frequency and presence across documents. This improved the model's ability to classify sentiment accurately.

## Conclusion
This project demonstrates the application of sentiment analysis on customer reviews using machine learning. By preprocessing the text data and utilizing the Naive Bayes classifier, we were able to extract valuable insights into customer satisfaction, helping businesses improve their services based on customer feedback.

## Future Improvements
- Experiment with other models like Support Vector Machines (SVM) or deep learning-based models.
- Fine-tune preprocessing to handle complex phrases better.
- Expand the dataset to include more reviews for a more comprehensive analysis.

## Technologies Used
- Python
- Scikit-learn
- NLTK
- Pandas
- NumPy

