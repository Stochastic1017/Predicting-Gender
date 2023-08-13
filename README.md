# Predicting Sex Based on First Names

Project Collaborators:
* Anais Corona Perez
* Shrivats Sudhir
* Yuqi Zhou
* Jason Lee
* Gyuho Shim

This project is intended to explain and report the supervised machine
learning models used to predict gender on the basis of first name. The data
includes 25,595 names with labels male, female, and uni-sex respectively.
Feature engineering was implemented to convert the names in the dataset
with special characters to ASCII equivalent, and reduce the dataset to
only account for binary targets male and female. One-hot encoding was
used to embed these names into 1-dimensional vectors. Finally, after
splitting the dataset into training, testing, and validating sets, we used
grid-search and gradient-boosting to evaluate and find the best model.

# About the Dataset
The website behind the name allows users to find the history and etymological
meaning of names in various languages. We used the **given names + gender**
dataset to train our model, which can be found [here](https://www.behindthename.com/api/download.php).

# Feature Engineering
We first removed all rows with labels **uni-sex**, as we were only concerned with binary targets **male** and **female**, which we labelled **0** and **1** respectively. Then, we mapped each name with ISO Latin 1 characters to the equivalent ASCII characters using the **unidecode** package (further details can be found [here](https://pypi.org/project/Unidecode)). This left us with 24,595 names to be used to train our data.

# One-Hot Encoding Names
Consider a $62$ length vector of the following form: 
```math
\begin{pmatrix} \text{a, b, ..., z} & | & \text{A, B, ..., Z} & | \text{ ', , AE, ae, Th, th, @, -, *, `} \end{pmatrix}
```
Each letter of a name can be one-hot encoded in the vector above. In order to make our one-hot encoding \textsf{sci-kit} usable, we pad enough rows with zeros so that the total number of rows is equals the length of the largest name in the dataset. In our data, the largest name  is $21$ characters long, i.e., each name in our dataset is encoded as a $(21 \times 62)$ 2-dimensional array, which after flattening gives us a 1-dimensional vector of size $1302$. An interactive-3d PCA projection can be found [here](https://htmlpreview.github.io/?https://github.com/Stochastic1017/Predicting_Gender/blob/main/PCA-3d.html).

# Algorithm to Find the Best Classifier
We used **GridSearchCV** with multicore **n_jobs = 5** to find the best hyper-parameters for binary classification using Knn and Logistic Regression. We found that **metric = cosine** and **n_neighbors = 3** were the best hyper-parameters for knn, and **C = 360** was the best hyper-parameter for logistic regression. Comparing validation accuracy's for both those models, along with decision tree classifier with **criterion = entropy** we found that decision tree classifier had the best performance.

# Gradient Boosting
As the validation accuracy for all our models is ranged from 0.71 to 0.74, we decided to fit an ensemble classifier model to maximize accuracy. As decision trees had the highest validation accuracy, we decided to go for the Gradient Boosting classifying model as it is an ensemble model that fits boosted decision trees by minimizing error gradient. After brute-forcing, we found that **n_estimators = 10000** and **learning_rate = 1.0** had the best validation accuracy at about 0.794, which was considerably better than the previous three models.

# Model Performance Assessment
After plotting Accuracy, Precision, Recall, and Area under the ROC curve, we can see that only Decision Trees and Gradient Boosting models have all scores over the threshold 0.7. Additionally, Gradient Boosting out performs every model by a decent margin, thus making it the final model that we choose for predicting gender based on names.

# Conclusion
In order to predict gender based on names, we used the \texttt{behind the name} data-set, conducted rigorous feature engineering, one-hot encoding, and splitting to make the data-set usable in \texttt{scikit}. After running a grid search and plotting the Knn, Logistic Regression, and Decision Trees with the most optimal hyper-parameters, we found that Decision trees had the best validation performance. Following this, we used gradient boosting ensemble classifier with 10,000 boosting stages and learning rate 1.0 to fit the best model that outperforms the previous three models by a considerable margin. 


