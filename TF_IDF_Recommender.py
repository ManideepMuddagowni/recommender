def TF_IDF_recommender(user_q,df_shop, u):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from Get_Recommendation import get_recommendation
    #fitting and transforming the vector
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_itemid = tfidf_vectorizer.fit_transform(df_shop['text']) #fitting and transforming the vector
    user_tfidf = tfidf_vectorizer.transform(user_q['text'])
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_itemid)
    output = list(cos_similarity_tfidf)
    top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:100]
    list_scores = [output[i][0][0] for i in top]
    return get_recommendation(top, list_scores,u,df_shop)