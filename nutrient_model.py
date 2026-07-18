import pandas as pd
from fastapi import FastAPI,APIRouter
from rapidfuzz import fuzz, process

app=FastAPI()

# Load data
df = pd.read_excel("food_600.xlsx")

df["food_clean"]=df["food_name"].str.lower().str.strip()

def get_food_nutrients(food_name):
    food_name=food_name.lower().strip()

    #exact match
    exact_match=df[df["food_clean"]==food_name]
    if not exact_match.empty:
        row=exact_match.iloc[0]
        return {
    "match_percentage": 100,
    "calories": float(row["calories"]),
    "protein": float(row["protein"]),
    "carbs": float(row["carbs"]),
    "fat": float(row["fat"])
}
    match=process.extractOne(
        food_name,
        df["food_clean"],
        scorer=fuzz.token_sort_ratio
    )

    if match is None:
        return {"message":"Food not found."}
    
    matched_name,score,Index=match

    if score<60:
        return {"message":"Food not found."}
    
    row=df.iloc[Index]

    return {
    "match_percentage": round(float(score), 2),
    "calories": float(row["calories"]),
    "protein": float(row["protein"]),
    "carbs": float(row["carbs"]),
    "fat": float(row["fat"])
}

router=APIRouter(prefix="/nutrition",tags=["Nutrition"])

@router.get("/{food_name}")
def get_nutrients(food_name:str):
    return get_food_nutrients(food_name)

app.include_router(router)