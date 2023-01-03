# Function that takes in item title as input and outputs most similar products
def get_recommendations(TITLE,Tags, cosine_sim,indices,df_shop):
    # Get the index of the items that matches the title
    idx = indices[TITLE]

    # Get the pairwsie similarity scores of all items
    sim_scores = list(enumerate(cosine_sim[idx])) # Sort the items
    #Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar items
    sim_scores = sim_scores[1:15]# Item indicies, Get the items indices
    shop_indices = [i[0] for i in sim_scores]
   
     
    # It reads the top 5 recommended Items titles and print the images
    #print(df_shop['name'].iloc[shop_indices])
    return df_shop[['sku','name','price']].iloc[shop_indices] 
    
    '''for i in rec['Images']:
        response = requests.get(i)
        img = Image.open(BytesIO(response.content))
        plt.figure()
        print(plt.imshow(img), )'''