# Function for removing the html tags
def remove_html(text):
    import re
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)