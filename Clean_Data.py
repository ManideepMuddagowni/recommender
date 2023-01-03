#Cleaning the the category and tags column
def clean_data(x):
        if isinstance(x, list): 
            return [str.lower(i.replace("|", ",")) for i in x]             
        else: 
            #Check if director exists. If not, return empty string          
            if isinstance(x, str):
                return str.lower(x.replace(">", ","))          
            else:           
                return '' 
            