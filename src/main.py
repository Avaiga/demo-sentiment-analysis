""" Creates a sentiment analysis App using Taipy"""
import pandas as pd
import numpy as np
from taipy.gui import Gui, notify

import requests
import os


API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
headers = {"Authorization": f"Bearer {os.getenv('API_SECRET')}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

text = "Original text"

page = """
# Getting started with **Taipy**{: .color-primary} **GUI**{: .color-primary}

<|layout|columns=1 1|
<|
**My text:** <|{text}|>

**Enter sentence(s):**
<|{text}|input|>
<|Analyze|button|on_action=local_callback|>
|>


<|Table|expandable|
<|{dataframe}|table|number_format=%.2f|>
|>
|>

<|1 1 1|layout|
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


def analyze_text(input_text: str) -> dict:
    """
    Runs the sentiment analysis model on the text

    Args:
        - text (str): text to be analyzed

    Returns:
        - dict: dictionary with the scores
    """
    outputs = query({
	"inputs": input_text,
    })
    print(outputs)

    scores = {"Text": input_text[:50]}

    for output in outputs[0]:
        if output["label"] == 'positive':
            scores["Score Pos"] = output["score"]
        elif output["label"] == 'neutral':
            scores["Score Neu"] = output["score"]
        elif output["label"] == 'negative':
            scores["Score Neg"] = output["score"]
    scores["Overall"] = (scores["Score Pos"] - scores["Score Neg"])

    return scores


def local_callback(state) -> None:
    """
    Analyze the text and updates the dataframe

    Args:
        - state: state of the Taipy App
    """
    notify(state, "Info", f"The text is: {state.text}", True)
    temp = state.dataframe.copy()
    scores = analyze_text(state.text)
    temp.loc[len(temp.index)] = scores
    state.dataframe = temp
    state.text = ""


path = ""
treatment = 0

page_file = """
<|{path}|file_selector|extensions=.txt|label=Upload .txt file|on_action=analyze_file|> <|{f'Downloading {treatment}%...'}|>

<br/>

<|Table|expandable|
<|{dataframe2}|table|width=100%|number_format=%.2f|>
|>

<br/>

<|{dataframe2}|chart|type=bar|x=Text|y[1]=Score Pos||y[2]=Score Neu|y[3]=Score Neg|y[4]=Overall|color[1]=green|color[2]=grey|color[3]=red|type[4]=line|height=600px|>

"""


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
        print(scores)
        temp.loc[len(temp.index)] = scores
        state.dataframe2 = temp

    state.path = None


pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "line": page,
    "text": page_file,
}


Gui(pages=pages).run(title="Sentiment Analysis", port=4083)
