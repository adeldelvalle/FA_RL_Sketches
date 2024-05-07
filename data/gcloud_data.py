from google.cloud import bigquery as bq
import os
from tqdm import tqdm

# directory to store
save_dir = "gc-data"

# From Google BigQuery  (https://pypi.org/project/google-cloud-bigquery/)
client = bq.Client()
secret_dir = "secret/"
api_key = cwd + "/" + secret_dir + os.listdir(secret_dir)[0]
assert api_key[-5:] == ".json"  # confirm that it was found
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key

# use api key for the session
def run_query(query_string):
    print("Running Query:")
    print(query_string)
    print()
    dataframe = (
        client.query(query_string)
        .result()
        .to_dataframe(
            create_bqstorage_client=True,
        )
    )
    print(dataframe.head())
    return dataframe

def query_daily(tables):
    limit = 50000

    # QUERYtemp = f"""
    #     SELECT
    #        T.date_local AS Day, AVG(T.arithmetic_mean) AS Temperature, AVG(first_max_value), AVG(first_max_hour), AVG(observation_count), AVG(observation_percent), ANY_VALUE(state_name), ANY_VALUE(county_name), ANY_VALUE(city_name)
    #     FROM
    #       `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary` as T

    #     GROUP BY Day
    #     ORDER BY Day

    #     LIMIT {limit}
    # """
    # # client = bigquery.Client.from_service_account_json('key.json')
    # df_temp = client.query(QUERYtemp).to_dataframe().set_index('Day')
    # print("Temp query attained...")

    # df_temp.index.name = 'Day'
    # df_temp.to_csv(f"{save_dir}/temp.csv")
    # print("Temp query saved")

    for table_name in tqdm(tables):
        print(table_name, "query starting")

        query_table = f"""
            SELECT
               o3.date_local AS Day, AVG(o3.aqi) AS o3_AQI, AVG(first_max_value), AVG(first_max_hour), AVG(observation_count), AVG(observation_percent), ANY_VALUE(state_name), ANY_VALUE(county_name), ANY_VALUE(city_name)
            FROM
              `bigquery-public-data.epa_historical_air_quality.{table_name}` as o3

            GROUP BY Day
            ORDER BY Day

            LIMIT {limit}
        """

        # client = bigquery.Client.from_service_account_json('key.json')
        df_temp = client.query(query_table).to_dataframe().set_index('Day')

        df_temp.index.name = 'Day'
        
        df_temp.to_csv(f"{save_dir}/{table_name}.csv")


sus_tables = ['codaily', 'pm25daily', 'no2daily', 'o3daily', 'temp']
converted_sus_tables = ['no2_daily_summary']
semi_sus_tables = ['o3_daily_summary', 'pm10_daily_summary', 'pm25_frm_daily_summary', 'pressure_daily_summary', 
                   'rh_and_dp_daily_summary', 'so2_daily_summary', 'voc_daily_summary', 'wind_daily_summary']
cited_tables = ['pm25_nonfrm_dail_summary', 'wind_daily_summary', 'temperature_daily_summary']

query_daily(converted_sus_tables)