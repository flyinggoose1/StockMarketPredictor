import os
import pandas as pd
from datetime import datetime

class DataFetcher:

    def __init__(self):
        pass
    
    def getHistoricalData(self, stock_symbol):
        # Define the directory and filename for the CSV file
        dir_name = "D:\\IBDP\\Extended Essay\\Stock Data(By Author)"
        filename = stock_symbol + ".csv"
        outputfile = os.path.join(dir_name, filename)
        
        # Check if the file exists
        if os.path.isfile(outputfile):
            return pd.read_csv(outputfile)  # Return the data as a DataFrame
        
        # If the file does not exist, raise an error
        raise FileNotFoundError(f"The file {outputfile} does not exist.")

# Example usage
'''if __name__ == "__main__":
    fetcher = DataFetcher()
    try:
        data = fetcher.getHistoricalData("AMZN")
        print(data.head())  # Print the first few rows of the DataFrame
    except FileNotFoundError as e:
        print(e)'''



