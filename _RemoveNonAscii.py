# Function for removing NonAscii characters in the description column
def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)



