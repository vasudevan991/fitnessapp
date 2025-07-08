import pandas as pd
import sqlite3

def load_usda_csv_to_db(csv_file):
    df = pd.read_csv(csv_file)

    # Rename columns (edit these based on your actual column names)
    df = df.rename(columns={
        "Description": "name",
        "Energy (kcal)": "calories",
        "Protein (g)": "protein",
        "Carbohydrate, by difference (g)": "carbs",
        "Total lipid (fat) (g)": "fat"
    })

    # Filter to essential columns
    df = df[["name", "calories", "protein", "carbs", "fat"]]

    conn = sqlite3.connect("usda_food.db")
    df.to_sql("foods", conn, if_exists="replace", index=False)
    conn.close()

    print("Database created with USDA food data.")
