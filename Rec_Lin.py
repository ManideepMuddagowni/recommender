def rec_lin(user_input, price_choice, linear):
    #extracted = process.extract("pen drive", choices, limit=10)
    #extracted
    price_choice=price_choice.astype(float)
    # use fuzzywuzzy to grab the product with name closest to user input
    extracted = process.extract(user_input, choices, limit=10)
    product_name=[]
    for i in range(10):
        product_name.insert(i,extracted[i][0])
    
    # Get the index of the product that matches the product name
    idx = indices[product_name]
    
    df_return = df_new[['sku',"name","price"]].loc[idx]
    df_return=df_return[df_return['price'] < (price_choice+(0.1*price_choice))]
    df_return=df_return[(price_choice-(0.1*price_choice))<df_return['price']]
    
    # Return the top 10 most similar products
    
    return df_return.sort_values(by="price")[['sku',"name","price"]]
    #return product_name