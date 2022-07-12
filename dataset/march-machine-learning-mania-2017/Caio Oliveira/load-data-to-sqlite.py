import sqlite3
import csv
import re

DATA_DIR = "../input"

CSV_NAMES = ["RegularSeasonCompactResults",
             "RegularSeasonDetailedResults",
             "Seasons",
             "Teams",
             "TourneyCompactResults",
             "TourneyDetailedResults",
             "TourneySeeds",
             "TourneySlots"]

STR_COLS = ['wloc', 'seed', 'strongseed', 'weakseed', 'regionw', 'regionx',
            'regiony', 'regionz', 'dayzero', 'team_name', 'slot']

DB_PATH = "data.db"

def connection():
    return sqlite3.connect(DB_PATH)

def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def name_csv_to_sql(csv_col):
    return re.sub('_+', '_', camel_to_snake(csv_col).replace(" ", "_"))

def create_table(tbl_name, cols, cursor):
    cursor.execute("DROP TABLE IF EXISTS %s;" % tbl_name)
    cursor.execute("CREATE TABLE %s (%s);" % (tbl_name, ", ".join(cols)))

def coerce_type(col, v):
    if col not in STR_COLS:
        v = int(v)
    return v

def prepare_data(tbl):
    return [{name_csv_to_sql(k): coerce_type(name_csv_to_sql(k), v)
             for k,v in row.items()}
            for row in tbl]

def load_csv(path):
    with open(DATA_DIR + path) as fd:
        return prepare_data(csv.DictReader(fd))

def csv_to_table(csv_name, cursor):
    tbl_name = name_csv_to_sql(csv_name)

    from_csv = load_csv("/%s.csv" % csv_name)
    cols = from_csv[0].keys()
    vs_for_db = [[row[k] for k in cols] for row in from_csv]
    insert_cmd = "INSERT INTO %s (%s) VALUES (%s)" % (tbl_name, ", ".join(cols), ", ".join((["?"] * len(cols))))

    create_table(tbl_name, cols, cursor)
    cursor.executemany(insert_cmd, vs_for_db)

def do_it(tables):
    with connection() as conn:
        cur = conn.cursor()
        for table in tables:
            csv_to_table(table, cur)
        conn.commit()


if __name__ == '__main__':
    do_it(CSV_NAMES)
