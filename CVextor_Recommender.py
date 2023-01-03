def CVextor_recommender(user_q, df_shop, u):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from Get_Recommendation import get_recommendation
    from sklearn.feature_extraction.text import CountVectorizer

    count_vectorizer = CountVectorizer()
    count_item_id = count_vectorizer.fit_transform((df_shop['text'])) #fitting and transforming the vector
    user_count = count_vectorizer.transform(user_q['text'])
    cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x),count_item_id)
    output = list(cos_similarity_countv)
    top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:100]
    list_scores = [output[i][0][0] for i in top]    
    return get_recommendation(top, list_scores,u,df_shop)