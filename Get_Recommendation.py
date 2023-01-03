def get_recommendation(top, scores,u,df_shop):
    import pandas as pd
    recommendation = pd.DataFrame(columns = ['User Id','sku','name','score'])
    count = 0
    for i in top:
        recommendation.at[count, 'User Id'] = u
        recommendation.at[count, 'sku_rec']= df_shop['sku'][i]  #Let's-test-the-recommender-by-selecting-the-user-with-item-Id-11574'] = df_shop['sku'][i]
        recommendation.at[count, 'name'] = df_shop['name'][i]
        recommendation.at[count, 'score'] =  scores[count]
        #recommendation.at[count, 'price'] = df_shop['price']
        count += 1
    return recommendation