import sqlite3

def display_tables_and_contents(database_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    # Get a list of all tables in the database
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = c.fetchall()

    # Display tables, their structure, and contents
    for table in tables:
        print(f"Table: {table[0]}")

        # Get the table schema
        c.execute(f"PRAGMA table_info({table[0]});")
        columns = c.fetchall()
        print("Columns:")
        for column in columns:
            print(f"  {column[1]}  {column[2]}")

        # Display table contents
        c.execute(f"SELECT * FROM {table[0]};")
        rows = c.fetchall()
        print("Data:")
        for row in rows:
            print(row)

    # Close the database connection
    conn.close()

# Replace 'users.db' with the actual path to your SQLite database
display_tables_and_contents('users.db')
