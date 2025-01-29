import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

import taipy.gui.builder as tgb
from taipy.gui import notify, Gui

# ----------------------------------
# Model & Data Initialization
# ----------------------------------
MODEL = "sbcBI/sentiment_analysis_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

text = "Original text"
dataframe = pd.DataFrame(
    {
        "Text": [""],
        "Score Pos": [0.33],
        "Score Neu": [0.33],
        "Score Neg": [0.33],
        "Overall": [0],
    }
)
dataframe2 = dataframe.copy()

path = ""
treatment = 0

# ----------------------------------
# Helper Functions
# ----------------------------------
def analyze_text(input_text: str) -> dict:
    """
    Runs the sentiment analysis model on the text.
    Returns a dictionary with the scores.
    """
    encoded_text = tokenizer(input_text, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    return {
        "Text": input_text[:50],
        "Score Pos": scores[2],
        "Score Neu": scores[1],
        "Score Neg": scores[0],
        "Overall": scores[2] - scores[0],
    }


def local_callback(state):
    """
    Analyze the text and update the main dataframe.
    """
    notify(state, "Info", f"The text is: {state.text}", True)
    temp = state.dataframe.copy()
    scores = analyze_text(state.text)
    temp.loc[len(temp)] = scores
    state.dataframe = temp
    state.text = ""


def analyze_file(state):
    """
    Analyze each line of the uploaded text file and update `dataframe2`.
    """
    state.dataframe2 = dataframe2
    state.treatment = 0

    with open(state.path, "r", encoding="utf-8") as f:
        data = f.read()
        file_list = data.split("\n")

    for i, input_text in enumerate(file_list):
        state.treatment = int((i + 1) * 100 / len(file_list))
        temp = state.dataframe2.copy()
        scores = analyze_text(input_text)
        temp.loc[len(temp)] = scores
        state.dataframe2 = temp

    state.path = None


# ----------------------------------
# Building Pages with TGB
# ----------------------------------

# ---------------------
# Home Page ("/")
# ---------------------
with tgb.Page() as root_page:
    tgb.toggle(theme=True)
    tgb.navbar()

# ---------------------
# "line" Page
# ---------------------
with tgb.Page() as page:
    tgb.text("# Getting started with **Taipy** GUI", mode="md")

    # Layout with two columns
    tgb.text("**My text:** {text}", mode="md")
    tgb.input("{text}", label="Enter a word:")
    tgb.button("Analyze", on_action=local_callback)

    # Display the main dataframe as a table
    tgb.text("### Analyzed Entries", mode="md")
    tgb.table("{dataframe}", number_format="%.2f")

    # Summaries in a 1-1-1 layout
    with tgb.layout(columns="1 1 1"):
        tgb.text(lambda dataframe: f"## Positive {np.mean(dataframe['Score Pos']):.2f}", class_name="h4")

        tgb.text(lambda dataframe: f" ## Neutral{np.mean(dataframe['Score Neu']):.2f}", class_name="h4")

        tgb.text(lambda dataframe: f" ## Negative{np.mean(dataframe['Score Neg']):.2f}", class_name="h4")

    # Bar + line chart
    tgb.chart(
        "{dataframe}",
        x="Text",
        y=["Score Pos", "Score Neu", "Score Neg", "Overall"],
        color=["green", "grey", "red", "yellow"],
        type=["bar", "bar", "bar", "line"],  # or pass a list for 'type'
        title="Sentiment Trends",
    )

# ---------------------
# "text" Page
# ---------------------
with tgb.Page() as page_file:
    tgb.text("## File Uploader", mode="md")
    # File selector & progress text
    tgb.file_selector(
        "{path}",
        label="Upload .txt file",
        extensions=".txt",
        on_action=analyze_file
    )
    tgb.text(lambda treatment: f"Downloading {treatment}%...", mode="md")

    tgb.text("### Analyzed Entries from File", mode="md")
    tgb.table("{dataframe2}", number_format="%.2f")

    # Bar + line chart for file-based results
    tgb.chart(
        "{dataframe2}",
        type="bar",
        x="Text",
        y=["Score Pos", "Score Neu", "Score Neg", "Overall"],
        color=["green", "grey", "red", None],
        subtypes={"Overall": "line"},
        title="Sentiment from File",
        height="600px",
    )

# ---------------------
# Run the App
# ---------------------
pages = {
    "/": root_page,
    "line": page,
    "text": page_file,
}
gui = Gui(pages=pages)
gui.run(title="Sentiment Analysis", dark_mode=True)

