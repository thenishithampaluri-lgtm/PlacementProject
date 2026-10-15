from flask import Flask, render_template
from models.load import load_data
from models.eda import summarize
from models.feature_engg import normalization
import pandas as pd
app = Flask(__name__)
@app.route("/")
def home():
    return load_page()
# Load Data Page
@app.route("/load")
def load_page():
    df = load_data()
    summary = summarize(df)
    return render_template("load.html",
                           shape=summary["shape"],
                           columns=summary["columns"]
    )

# EDA Page

@app.route("/eda")
def eda_page():

    df = load_data()
    summary = summarize(df)

    return render_template(
        "eda.html",
        shape=summary["shape"],
        columns=summary["columns"],
        missing_values=summary["missing_values"]
    )
@app.route("/feature_engg")
def feature_engg_page():
    df = load_data()
    train_df, test_df=normalization(df.select_dtypes('number'))
    return render_template(
        "feature_engg.html",
        columns=train_df.columns,
        train_rows=train_df.values,test_rows=test_df
    )
if __name__== "__main__":
    app.run(debug=True, port=8000)
