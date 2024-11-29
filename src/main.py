""" Creates a sentiment analysis App using Taipy"""
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

import numpy as np
import pandas as pd
from taipy.gui import Gui, notify

MODEL = "sbcBI/sentiment_analysis_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def analyze_text(input_text: str) -> dict:
    """
    Runs the sentiment analysis model on the text

    Args:
        - text (str): text to be analyzed

    Returns:
        - dict: dictionary with the scores
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


def local_callback(state) -> None:
    """
    Analyze the text and updates the dataframe

    Args:
        - state: state of the Taipy App
    """
    notify(state, "Info", f"The text is: {state.text}", True)
    temp = state.dataframe.copy()
    scores = analyze_text(state.text)
    temp.loc[len(temp)] = scores
    state.dataframe = temp
    state.text = ""


def analyze_file(state) -> None:
    """
    Analyse the lines in a text file

    Args:
        - state: state of the Taipy App
    """
    state.dataframe2 = dataframe2
    state.treatment = 0
    with open(state.path, "r", encoding="utf-8") as f:
        data = f.read()
        print(data)

        file_list = list(data.split("\n"))

    for i, input_text in enumerate(file_list):
        state.treatment = int((i + 1) * 100 / len(file_list))
        temp = state.dataframe2.copy()
        scores = analyze_text(input_text)
        temp.loc[len(temp)] = scores
        state.dataframe2 = temp

    state.path = None


if __name__ == "__main__":
    text = "Original text"

    page = """
# Getting started with **Taipy**{: .color-primary} **GUI**{: .color-primary}

<|layout|columns=1 1|
<|
**My text:** <|{text}|>

**Enter a word:**
<|{text}|input|>
<|Analyze|button|on_action=local_callback|>
|>


<|Table|expandable|
<|{dataframe}|table|width=100%|number_format=%.2f|>
|>
|>

<|layout|columns=1 1 1|
## Positive <|{np.mean(dataframe['Score Pos'])}|text|format=%.2f|raw|>

## Neutral <|{np.mean(dataframe['Score Neu'])}|text|format=%.2f|raw|>

## Negative <|{np.mean(dataframe['Score Neg'])}|text|format=%.2f|raw|>
|>

<|{dataframe}|chart|type=bar|x=Text|y[1]=Score Pos|y[2]=Score Neu|y[3]=Score Neg|y[4]=Overall|color[1]=green|color[2]=grey|color[3]=red|type[4]=line|>
    """

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

    page_file = """
<|{path}|file_selector|extensions=.txt|label=Upload .txt file|on_action=analyze_file|> <|{f'Downloading {treatment}%...'}|>

<br/>

<|Table|expandable|
<|{dataframe2}|table|width=100%|number_format=%.2f|>
|>

<br/>

<|{dataframe2}|chart|type=bar|x=Text|y[1]=Score Pos|y[2]=Score Neu|y[3]=Score Neg|y[4]=Overall|color[1]=green|color[2]=grey|color[3]=red|type[4]=line|height=600px|>
    """

    pages = {
        "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
        "line": page,
        "text": page_file,
    }

    Gui(pages=pages).run(title="Sentiment Analysis")
