import pickle
import pandas as pd
from utils import preprocess

# Load the saved pickle files
with open('user_final_rating.pkl', 'rb') as f:
    user_final_rating = pickle.load(f)

with open('reco_df.pkl', 'rb') as f:
    reco_df = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('sentiment_classifier.pkl', 'rb') as f:
    sentiment_classifier = pickle.load(f)

# Combining both pipelines: recommendation + sentiment analysis
def sentiment_recommender(user_id):
     # First check if user exists in the dataset
    if user_id not in user_final_rating.index:
        return None
    
    # Step 1: Recommend Top 20 Products
    user_input = user_id
    search = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
    search = search.reset_index()
    search.columns = ['id', 'Predicted_Score']
    
    main_df_unique = reco_df.drop_duplicates(subset='id')
    merged_df = pd.merge(
        search,
        main_df_unique[['id', 'name', 'reviews_rating', 'reviews_username', 'reviews_text']],
        on='id',
        how='left'
    )
    
    # Step 2: Analyze Sentiments for each product
    sentiment_scores = []
    
    for idx, row in merged_df.iterrows():
        review_text = str(row['reviews_text'])
        
        # Preprocess the review
        cleaned_review = preprocess(review_text, stem=False)
        
        # Vectorize
        X_review = vectorizer.transform([cleaned_review])
        
        # Predict sentiment
        predicted_sentiment = sentiment_classifier.predict(X_review)[0]
        
        # Assign 1 for Positive, 0 for Negative
        sentiment_score = 1 if predicted_sentiment == 'Positive' else 0
        sentiment_scores.append(sentiment_score)
    
    # Add sentiment score to merged_df
    merged_df['Sentiment_Score'] = sentiment_scores

    # Step 3: Select Top 5 Products
    final_recommendations = merged_df.sort_values(
        by=['Sentiment_Score', 'Predicted_Score'],
        ascending=[False, False]
    ).head(5)
    
    return final_recommendations[['name']]