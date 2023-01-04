import csv
import psycopg2
from pathlib import Path
from loguru import logger


DATA_PATH = Path(__file__).resolve().parent / "user_to_matches.csv"

DATABASE = "postgres"
USER = "postgres"
PASSWORD = "password"
HOST = "127.0.0.1"
PORT = "5432"
CREATE_TABLE_SQL = '''CREATE TABLE "user_to_matches" ("user_id" varchar, "match_id" varchar, "game" varchar, "created_at" timestamp, "membership" varchar, "faction" varchar, "winner" varchar);'''


def main(cursor:psycopg2.extensions.cursor) -> None:
   _create_table(cursor)
   logger.success("Table created successfully.")
   _import_data_into_table(cursor)
   logger.success("Data successfully imported.")


def _create_table(cursor:psycopg2.extensions.cursor) -> None:
   cursor.execute(CREATE_TABLE_SQL)


def _import_data_into_table(cursor:psycopg2.extensions.cursor) -> None:
   with open(DATA_PATH, 'r') as f:
      reader = csv.reader(f)
      next(reader) # Skip the header row.
      for row in reader:
         cursor.execute(
         "INSERT INTO user_to_matches VALUES (%s, %s, %s, %s, %s, %s, %s)",
         row
      )
   conn.commit()


if __name__ == "__main__":
   #establishing the connection
   conn = psycopg2.connect(
      database=DATABASE, user=USER, password=PASSWORD, host=HOST, port=PORT
   )
   conn.autocommit = True
   #Creating a cursor object using the cursor() method
   cursor = conn.cursor()
   main(cursor)
   conn.close()
