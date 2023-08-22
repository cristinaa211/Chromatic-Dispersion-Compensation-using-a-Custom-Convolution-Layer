import psycopg2
import json
import numpy as np

def read_table_postgresql(columns = None, table_name = None, database_config = None, 
                          limit = None, query = None):
    conn = psycopg2.connect(
        host=database_config['host'],
        port=database_config['port'],
        dbname=database_config['dbname'],
        user=database_config['user'],
        password=database_config['password']
    )    
    cursor = conn.cursor()
    if query == None:
        if limit : query = 'SELECT {} FROM PUBLIC."{}" LIMIT {}'.format(columns, table_name, int(limit))
        else: query = 'SELECT {} FROM PUBLIC."{}"'.format(columns, table_name)
    cursor.execute(query)
    data = cursor.fetchall()
    headers = [i[0] for i in cursor.description]
    print('Data fetched successfully')
    if conn:
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
    return headers, data


def  import_data_to_postgresql(data, table_name, database_config):
    """Imports data into table_name in the database
    data            : a list of lists containing label, detailed label and the processed radar sample values
    table_name      : a string reffering to the table name in the database
    database_config : the database configuration """
    conn = psycopg2.connect(
        host=database_config['host'],
        port=database_config['port'],
        dbname=database_config['dbname'],
        user=database_config['user'],
        password=database_config['password']
    )
    query_table = f"""CREATE TABLE IF NOT EXISTS Public.{table_name}(
    id SERIAL PRIMARY KEY,
    M integer,
    snr integer,
    input_data JSONB,
    input_net_real JSONB,
    input_net_imag JSONB,
    targets_real JSONB,
    targets_imag JSONB
    )"""
    query = f"""INSERT INTO {table_name} 
            (M, snr, input_data, input_net_real, input_net_imag, targets_real, targets_imag)
            VALUES (%s, %s,%s, %s, %s, %s, %s)"""
    try:
        with conn.cursor() as cur:
            cur.execute(query_table)
            for record in data:
                cur.execute(query, ( record['M'], record['snr'], 
                                     json.dumps(record['input_data']),
                                     json.dumps(record['input_net_real']), json.dumps(record['input_net_imag']),
                                     json.dumps(record['targets_real']), json.dumps(record['targets_imag'])))
            conn.commit()
        print("Data inserted successfully!")
    except psycopg2.Error as e:
        print(f"Error: {e}")
    finally:
        cur.close()
        conn.close()

        
def extract_signal_db(table_name, column, database_config):
    query = f"""SELECT {table_name} FROM PUBLIC.{column}  LIMIT 1;"""
    headers, data = read_table_postgresql(table_name=table_name,database_config= database_config, limit = 1, query = query)
    return headers, data


