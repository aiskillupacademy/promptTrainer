import dspy
from dspy.teleprompt import *
import os
import pandas as pd
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

gpt3_turbo = dspy.OpenAI(model='gpt-3.5-turbo-0125', max_tokens=1000)
dspy.configure(lm=gpt3_turbo)

st.title("Prompt Trainer")

def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question, voice -> answer")
    
    def forward(self, question, voice):
        return self.prog(question=question, voice=voice)

uploaded_file = st.file_uploader("Choose a CSV file")
input = st.text_input("Ask the question")
voice = st.selectbox("Selece Voice", ["PAS","FAB","AIDA","QUEST","BAB"])

if st.button("Enter"):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.dropna(inplace=True)
        df.reset_index(drop=True,inplace=True)
    else:
        df = pd.read_csv("C:/Users/Subham/llmteam/rava-llm-team-workflows/Fine tuning dataset - Sheet1.csv")
    qa_pair = []
    for i in range(df.shape[0]):
        qa_pair.append(dspy.Example(question=df["Input"][i], voice=df["Framework"][i], answer=df["Output"][i]).with_inputs("question", "voice"))
    with st.spinner():
        teleprompter = BootstrapFewShot(metric=validate_answer)
        optimized_program = teleprompter.compile(CoT(), trainset=qa_pair)
        pred = optimized_program(input, voice)
        original_string = pred.answer
        updated_string = original_string.replace("\\n", "\n")
        st.header("Predicted answer:")
        st.write(updated_string)
        st.header("Answer without training:")
        st.write(gpt3_turbo(input))

