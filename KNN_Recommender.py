

# Function that takes in item title as input and outputs most similar products
def KNN_recommender(user_q, df_shop, u ):
    from sklearn.neighbors import NearestNeighbors
    from Get_Recommendation import get_recommendation
    from sklearn.feature_extraction.text import TfidfVectorizer

    #n_neighbors = 11
    KNN = NearestNeighbors(n_neighbors=11)
    u=u
    df_shop=df_shop
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_itemid = tfidf_vectorizer.fit_transform((df_shop['text'])) #fitting and transforming the vector
    user_tfidf = tfidf_vectorizer.transform(user_q['text'])
    KNN.fit(tfidf_itemid)
    NNs = KNN.kneighbors(user_tfidf, return_distance=True) 
    top = NNs[1][0][1:]
    index_score = NNs[0][0][1:]
    return get_recommendation(top,  index_score, u, df_shop)