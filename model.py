import pickle
import numpy as np
from flask import jsonify


class SentimentBasedRecommender:
    def __init__(self, model_path, vectorizer_path, recommendation_model_path, dataset_path):
        # Load sentiment analysis model
        with open(model_path, 'rb') as model_file:
            self.sentiment_model = pickle.load(model_file)

        # Load TF-IDF vectorizer
        with open(vectorizer_path, 'rb') as vectorizer_file:
            self.tfidf_vectorizer = pickle.load(vectorizer_file)

        # Load user recommendation model
        with open(recommendation_model_path, 'rb') as rec_model_file:
            self.user_final_rating = pickle.load(rec_model_file)

        # Load user dataset
        with open(dataset_path, 'rb') as dataset_file:
            self.df = pickle.load(dataset_file)

    def get_recommendations(self, user_id, num_recommendations=5):
        # Perform sentiment-based recommendations
        sentiment_recommendations = self.get_sentiment_recommendations(
            user_id, num_recommendations)

        # Convert DataFrame to JSON-compatible dictionary
        sentiment_recommendations_dict = sentiment_recommendations.to_dict(
            orient='records')

        return jsonify(sentiment_recommendations_dict)

    def get_sentiment_recommendations(self, user_id, num_recommendations):
        recommendations = self.get_sentiment_recommendations_by_user(user_id)[
            :num_recommendations]
        return recommendations

    def get_sentiment_recommendations_by_user(self, user):
        if user in self.user_final_rating.index:
            # Get the product recommendations using the trained recommender system
            recommendations = list(
                self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

            # Filter the recommendations using the sentiment ML model
            temp = self.df[self.df.id.isin(recommendations)]

            # Transform the input data using saved tf-idf vectorizer
            X = self.tfidf_vectorizer.transform(
                temp["cleaned_reviews"].values.astype(str))

            # Predict sentiment using the trained sentiment model
            temp["predicted_sentiment"] = self.sentiment_model.predict(X)

            # Extract relevant columns for analysis
            temp = temp[['name', 'predicted_sentiment']]

            # Group by product name and calculate sentiment metrics
            temp_grouped = temp.groupby('name', as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.name.apply(lambda x: temp[(
                temp.name == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
            temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"] / temp_grouped["total_review_count"] * 100, 2)

            # Sort the recommendations by positive sentiment percentage
            return temp_grouped.sort_values('pos_sentiment_percent', ascending=False)
        else:
            return f"User name '{user}' doesn't exist"
