import streamlit as st
import pandas as pd
import datetime
from sqlalchemy import create_engine
import os
import sqlite3
import requests
from sqlalchemy import inspect, text

# --- Database Setup ---
DB_FILE = 'calorie_data.db'
engine = create_engine(f'sqlite:///{DB_FILE}')

# --- Helper Functions ---
def import_csv_to_sqlite(csv_file='calorie_data.csv'):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, parse_dates=['date'])
        df.to_sql('calorie_log', engine, if_exists='replace', index=False)
        return True
    return False

def load_data():
    try:
        return pd.read_sql('SELECT * FROM calorie_log', engine, parse_dates=['date'])
    except Exception:
        return pd.DataFrame(columns=['date', 'meal', 'food', 'quantity', 'calories', 'protein', 'carbs', 'fat'])

def save_data(df):
    df.to_sql('calorie_log', engine, if_exists='replace', index=False)

def calculate_calories(food_data, grams):
    cal = food_data[1] * grams / 100
    protein = food_data[2] * grams / 100
    carbs = food_data[3] * grams / 100
    fat = food_data[4] * grams / 100
    return cal, protein, carbs, fat

def calculate_per_quantity(nutriments, grams):
    factor = grams / 100
    return {
        "calories": round(nutriments.get("energy-kcal_100g", 0) * factor, 2),
        "protein": round(nutriments.get("proteins_100g", 0) * factor, 2),
        "carbs": round(nutriments.get("carbohydrates_100g", 0) * factor, 2),
        "fat": round(nutriments.get("fat_100g", 0) * factor, 2),
    }

# --- Streamlit App ---
st.set_page_config(page_title="Calorie Counter", layout="centered")
st.title("🍎 Calorie Counter App")

# Sidebar: Set daily goal
daily_goal = st.sidebar.number_input("Set your daily calorie goal", min_value=1000, max_value=5000, value=2000, step=50)

# Import CSV to SQLite (one-time or on demand)
if st.sidebar.button("Import CSV to SQLite"):
    if import_csv_to_sqlite():
        st.sidebar.success("CSV imported to SQLite database!")
    else:
        st.sidebar.error("CSV file not found.")

# Load data
data = load_data()
today = datetime.date.today()

# --- Add Meal ---
st.header("Add a Meal")
st.info("To add food, please use the search below.")

# --- Image Upload ---
st.subheader("Upload Image (JPG/PNG)")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

def open_food_facts_search(query):
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 5
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data.get("products", [])
    except Exception as e:
        st.error(f"Open Food Facts error: {e}")
        return []

def display_off_results():
    st.subheader("Search Your Bite")
    query = st.text_input("Search food (Open Food Facts):")
    if not query:
        return

    results = open_food_facts_search(query)

    if not results:
        st.warning("No foods found.")
        return

    product_options = [f"{p.get('product_name', 'Unnamed')} ({p.get('brands', 'Unknown')})" for p in results]
    selected_option = st.selectbox("Select food", product_options)

    selected_product = results[product_options.index(selected_option)]
    n = selected_product.get("nutriments", {})
    grams = st.number_input("Quantity (grams)", value=100, step=1)

    def safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    calories = round(safe_float(n.get("energy-kcal_100g", 0)) * grams / 100, 2)
    protein = round(safe_float(n.get("proteins_100g", 0)) * grams / 100, 2)
    carbs = round(safe_float(n.get("carbohydrates_100g", 0)) * grams / 100, 2)
    fat = round(safe_float(n.get("fat_100g", 0)) * grams / 100, 2)

    st.metric("Calories", f"{calories} kcal")
    st.metric("Protein", f"{protein} g")
    st.metric("Carbs", f"{carbs} g")
    st.metric("Fat", f"{fat} g")

    with st.form("add_off_meal"):
        meal = st.selectbox("Meal", ["Breakfast", "Lunch", "Dinner", "Snack"])
        submit = st.form_submit_button("Add to Meal")
        if submit:
            entry = {
                'date': datetime.date.today(),
                'meal': meal,
                'food': selected_product.get('product_name', 'Unnamed'),
                'quantity': grams,
                'calories': calories,
                'protein': protein,
                'carbs': carbs,
                'fat': fat
            }
            df = load_data()
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            save_data(df)
            st.success(f"✅ Added {grams}g of {entry['food']} to {meal}")

display_off_results()

# --- Today's Summary ---
st.header("Today's Summary")
today_data = data[data['date'] == pd.Timestamp(today)]
total_cal = today_data['calories'].sum()
total_protein = today_data['protein'].sum()
total_carbs = today_data['carbs'].sum()
total_fat = today_data['fat'].sum()

col1, col2 = st.columns(2)
col1.metric("Calories Consumed", f"{int(total_cal)} kcal")
col2.metric("Goal", f"{daily_goal} kcal")
st.progress(min(total_cal / daily_goal, 1.0))

# Macronutrient breakdown
st.subheader("Macronutrient Breakdown")
st.write(f"Protein: {total_protein:.1f}g | Carbs: {total_carbs:.1f}g | Fat: {total_fat:.1f}g")
st.plotly_chart({
    "data": [{
        "values": [total_protein, total_carbs, total_fat],
        "labels": ["Protein", "Carbs", "Fat"],
        "type": "pie"
    }],
    "layout": {"title": "Macronutrient Breakdown"}
}, use_container_width=True)

# --- Clear Today's Meals ---
def clear_today_meals():
    data = load_data()
    today = datetime.date.today()
    # Remove today's meals
    data = data[data['date'] != pd.Timestamp(today)]
    save_data(data)
    st.success("Today's meals have been cleared.")

# --- Meal Log ---
st.subheader("Today's Meal Log")
if not today_data.empty:
    today_data = today_data.copy()
    # Add buttons to filter by meal type
    meal_types = ["Breakfast", "Lunch", "Dinner", "Snack"]
    selected_meal = st.radio("Show meals for:", meal_types + ["All"], index=len(meal_types))
    if selected_meal != "All":
        filtered_data = today_data[today_data['meal'] == selected_meal]
    else:
        filtered_data = today_data
    # Calculate totals for the filtered data
    total_row = pd.DataFrame({
        'meal': ['Total'],
        'food': [''],
        'quantity': [filtered_data['quantity'].sum()],
        'calories': [filtered_data['calories'].sum()],
        'protein': [filtered_data['protein'].sum()],
        'carbs': [filtered_data['carbs'].sum()],
        'fat': [filtered_data['fat'].sum()]
    })
    display_df = pd.concat([filtered_data[['meal', 'food', 'quantity', 'calories', 'protein', 'carbs', 'fat']], total_row], ignore_index=True)
    st.dataframe(display_df)
    if st.button("Clear Today's Meals"):
        clear_today_meals()
        st.experimental_rerun()
else:
    st.info("No meals logged for today.")

# --- Yesterday's Total Calories ---
yesterday = today - datetime.timedelta(days=1)
yesterday_data = data[data['date'] == pd.Timestamp(yesterday)]
yesterday_total_cal = yesterday_data['calories'].sum()
st.sidebar.markdown(f"**Yesterday's Calories:** {int(yesterday_total_cal)} kcal")

# --- Profile Section ---
st.sidebar.header("Profile")
current_weight = st.sidebar.number_input("Current Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
target_weight = st.sidebar.number_input("Target Weight (kg)", min_value=30.0, max_value=300.0, value=65.0, step=0.1)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)

# Water tracking
st.sidebar.markdown("---")
st.sidebar.subheader("Water Tracker")
target_water_ml = 2000  # 2 liters
if "water_today" not in st.session_state:
    st.session_state["water_today"] = 0
if st.sidebar.button("Add 250ml Water"):
    st.session_state["water_today"] += 250
if st.sidebar.button("Add 100ml Water"):
    st.session_state["water_today"] += 100
if st.sidebar.button("Reset Water"):
    st.session_state["water_today"] = 0
st.sidebar.progress(min(st.session_state["water_today"] / target_water_ml, 1.0), text=f"{st.session_state['water_today']} ml / {target_water_ml} ml")

# Calculate BMI
bmi = current_weight / ((height / 100) ** 2)
st.sidebar.metric("BMI", f"{bmi:.1f}")

# --- History ---
st.header("History")
if st.checkbox("Show history"):
    st.dataframe(data.sort_values('date', ascending=False))
    st.line_chart(data.groupby('date')['calories'].sum())

# --- Export/Import ---
st.sidebar.header("Data Management")
if st.sidebar.button("Export Data as CSV"):
    st.sidebar.download_button(
        label="Download CSV",
        data=data.to_csv(index=False),
        file_name='calorie_data_export.csv',
        mime='text/csv'
    )
if st.sidebar.button("Clear All Data"):
    if st.sidebar.confirm("Are you sure you want to clear all data?"):
        data = data.iloc[0:0]
        save_data(data)
        st.sidebar.success("All data cleared.")

# --- Fitness Vault ---
st.header("🏋️‍♂️ Fitness Vault")

# Fitness categories and example routines
yoga_routines = ["Sun Salutation", "Vinyasa Flow", "Yin Yoga", "Power Yoga"]
hiit_routines = ["Tabata", "EMOM 20min", "HIIT Cardio", "Bodyweight Blast"]
strength_routines = ["Full Body Strength", "Upper Body", "Lower Body", "Core Strength"]
cardio_routines = ["Running 5km", "Cycling 30min", "Jump Rope", "Rowing 20min"]

fitness_categories = {
    "Yoga": yoga_routines,
    "HIIT": hiit_routines,
    "Strength": strength_routines,
    "Cardio": cardio_routines
}

# --- Fitness Log DB Setup ---
FITNESS_DB_FILE = DB_FILE  # Use same DB
FITNESS_TABLE = 'fitness_log'

def create_fitness_table():
    insp = inspect(engine)
    if not insp.has_table(FITNESS_TABLE):
        with engine.connect() as conn:
            conn.execute(text(f'''
                CREATE TABLE IF NOT EXISTS {FITNESS_TABLE} (
                    date TEXT,
                    category TEXT,
                    routine TEXT,
                    completed INTEGER
                )
            '''))
create_fitness_table()

def load_fitness_log():
    try:
        return pd.read_sql(f'SELECT * FROM {FITNESS_TABLE}', engine, parse_dates=['date'])
    except Exception:
        return

def save_fitness_log(df):
    df.to_sql(FITNESS_TABLE, engine, if_exists='replace', index=False)

# --- Fitness Vault UI ---
st.subheader("Today's Fitness Routine")
fitness_log = load_fitness_log()

category = st.selectbox("Select Category", list(fitness_categories.keys()))
routine = st.selectbox("Select Routine", fitness_categories[category])

# Check if already completed today
completed_today = False
if not fitness_log.empty:
    today_fitness = fitness_log[(fitness_log['date'] == str(today)) & (fitness_log['category'] == category) & (fitness_log['routine'] == routine)]
    completed_today = not today_fitness[today_fitness['completed'] == 1].empty

if completed_today:
    st.success(f"You have completed '{routine}' in {category} today!")
else:
    if st.button("Mark as Completed"):
        new_entry = pd.DataFrame([{ 'date': str(today), 'category': category, 'routine': routine, 'completed': 1 }])
        fitness_log = pd.concat([fitness_log, new_entry], ignore_index=True)
        save_fitness_log(fitness_log)
        st.success(f"Marked '{routine}' as completed for today!")
        st.experimental_rerun()

# Show progress for this week
st.subheader("Weekly Progress")
week_dates = [today - datetime.timedelta(days=i) for i in range(6, -1, -1)]
progress = []
for d in week_dates:
    done = (
        not fitness_log[(fitness_log['date'] == str(d)) & (fitness_log['category'] == category) & (fitness_log['routine'] == routine) & (fitness_log['completed'] == 1)].empty
    )
    progress.append(1 if done else 0)
progress_percent = sum(progress) / 7
st.progress(progress_percent, text=f"{sum(progress)}/7 days completed")
st.write({d.strftime('%a'): ('✅' if p else '❌') for d, p in zip(week_dates, progress)})
