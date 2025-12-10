# Movie Recommendation System: Collaborative Filtering vs. Machine Learning

**Author:** Nakul Magotra  
**Course:** CSCI 6970 – Machine Learning  
**Project:** Final Course Project – MS in Computer Science

## 📌 Project Overview
This project implements and compares multiple recommendation system architectures to predict user movie ratings. Moving beyond simple similarity matching, this study evaluates traditional **Collaborative Filtering** techniques against a **Feature-Based Machine Learning** approach using Gradient Boosting.

The goal is to solve the "choice paralysis" problem by accurately predicting how a user will rate a specific movie they haven't seen yet, using the **MovieLens 20M** dataset.

## 📄 Key Files
* `MovieRecommendation_CF_ML.ipynb`: The complete Jupyter Notebook containing data preprocessing, model implementation, training pipeline, and evaluation.
* `Report.pdf`: A detailed final report analyzing the methodology, mathematical foundations, and experimental results.

## 🚀 Models Implemented
The project compares four distinct approaches:

1.  **Item-based k-Nearest Neighbors (Item-kNN):** Predicts ratings based on item-item similarity using cosine distance.
2.  **User-based k-Nearest Neighbors (User-kNN):** Predicts ratings based on user-user similarity.
3.  **Matrix Factorization (SVD):** Uses Truncated SVD to reduce dimensionality and find latent factors connecting users and items.
4.  **Feature-Based Machine Learning (XGBoost):** A regression approach that treats recommendation as a supervised learning task. Features engineered include:
    * Global Bias
    * User Mean Rating
    * Movie Mean Rating
    * Normalized Timestamp

## 📊 Results
The models were evaluated using **RMSE** (Root Mean Squared Error) and **MAPE** (Mean Absolute Percentage Error) on a time-based leave-one-out train/test split.

| Model | RMSE | MAPE (%) | Performance |
| :--- | :--- | :--- | :--- |
| **XGBoost (Feature-Based)** | **0.9301** | **30.20%** | 🏆 **Best Performer** |
| Item-kNN | 0.9723 | 30.67% | 🥈 Runner Up |
| User-kNN | 1.0211 | 31.67% | |
| SVD (Matrix Factorization) | 3.1021 | 76.21% | Needs Optimization |

> **Analysis:** The feature-based XGBoost model outperformed traditional collaborative filtering by capturing specific user biases (e.g., critical vs. lenient raters) and global item popularity trends.

## 📂 Dataset
This project uses the **MovieLens 20M Dataset**, which contains 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users.

**Download the data here:**
[https://grouplens.org/datasets/movielens/20m/](https://grouplens.org/datasets/movielens/20m/)

*Note: The dataset is not included in this repository due to size constraints. Please download `ratings.csv` and `movies.csv` and place them in the project directory.*

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/movie-recommendation-system.git](https://github.com/your-username/movie-recommendation-system.git)
    cd movie-recommendation-system
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn jupyter
    ```

3.  **Run the Notebook:**
    ```bash
    jupyter notebook MovieRecommendation_CF_ML.ipynb
    ```

4.  **Configuration:**
    Ensure you update the `data_dir` path in the notebook configuration cell to point to your downloaded MovieLens CSV files.

## 🔮 Future Improvements
* **Neural Collaborative Filtering:** Replacing the linear SVD component with a Neural Network (using PyTorch/TensorFlow) to better capture non-linear user-item interactions.
* **Hybridization:** Creating a weighted ensemble of the Item-kNN and XGBoost predictions.
* **Cold Start Handling:** Integrating movie metadata (genres, tags) more deeply to handle new movies with zero ratings.
