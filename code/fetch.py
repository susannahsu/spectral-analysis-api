from astroquery.sdss import SDSS
import pandas as pd 

class SDSSDataFetcher:
    def __init__(self):
        pass

    def fetch_by_adql(self, adql_query):
        """
        Fetch data from SDSS using an ADQL query.

        input: ADQL query string
        """
        try:
            return SDSS.query_sql(adql_query)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def fetch_by_constraints(self, table_name, num, constraints):
        """
        Fetch data from SDSS based on constraints.

        input: {constraint variable: constraint value}
        e.g. {'ra': '<10', 'dec': '>0'}
        
        """
        adql_query = self._construct_query_from_constraints(table_name, num, constraints)
        return self.fetch_by_adql(adql_query)

    def _construct_query_from_constraints(self, table_name, num, constraints):
        """
        Construct an ADQL query string from a constraints dictionary.
        """
        where_clause = ' AND '.join([f"{key} {value}" for key, value in constraints.items()])
        query = f"SELECT TOP {num} * FROM {table_name} WHERE {where_clause}"
        return query

    def process_sdss_format_data(self, data):
        """
        Process data already in SDSS format.
        input: data in csv format.
        """
        return pd.read_csv(data)

# Example usage:
# sdss_fetcher = SDSSDataFetcher()

#For the fetch_by_adql method, the input is an ADQL query string.
# adql_query = "SELECT TOP 10 * FROM SpecObj"
# results = sdss_fetcher.fetch_by_adql(adql_query)
# print(results)


#And for the constraints method, the input is a dictionary of constraints along with the table name and number of results.
# constraints = {'ra': '<10', 'dec': '>0'}
# table = 'SpecObj'
# num = 10
# results = sdss_fetcher.fetch_by_constraints(table, num, constraints)
# print(results)
