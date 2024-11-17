import numpy as np
import pandas as pd
import json


# Load the JSON data
with open('/Users/tenzindhonden/Desktop/CS311/cs311-ai-project/tracks.json') as f:
    data = json.load(f)

# Convert JSON data to a pandas DataFrame
df = pd.DataFrame(data)
print(df.columns)
    
    