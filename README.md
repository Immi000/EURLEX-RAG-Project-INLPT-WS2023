# NLP_and_Transformers
Group project and assignments of the NLP and Transformer course

## Installation & Usage Instruction

### Clone the repository

To clone the repository, navigate to the location you want to save it and execute

```
git clone https://github.com/Immi000/NLP_and_Transformers-INLPT-WS2023
```
or alternatively
```
git clone ssh://git@github.com/Immi000/NLP_and_Transformers-INLPT-WS2023.git
```

### Execute the Chatmodel

To execute the chatmodel, you need to first install the required packages with

```
pip install -Ur requirements.txt
```


If you want to execute the Chat UI in Standalone Mode, execute the following command in the root directory of the project:

```
python3 standalone_ui.py
```
This will start a local instance of a streamlit server and a browser window should open, displaying the UI of the chatmodel. If the window does not open automatically, you can simply put http://localhost:8501 into the address bar of you browser to open it manually.

If you want to execute the Chat UI inside a Jupyter Notebook, open the file ```main.ipynb``` and execute the first two cells.

By default, logging is turned on for the standalone UI. While the streamlit server runs, the chatmodel will log every step it performs to the console. This feature is turned off in the notebook to avoid cluttering the screen.