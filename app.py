import dspy
from dspy.teleprompt import *
import os
import pandas as pd
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

gpt3_turbo = dspy.OpenAI(model='gpt-3.5-turbo-0125', max_tokens=1000)
dspy.configure(lm=gpt3_turbo)

st.title("Prompt Trainer")

def validate_hops(example, pred, trace=None):
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

uploaded_file = st.file_uploader("Choose a CSV file")
input = st.text_input("Ask the question")
if st.button("Enter"):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("C:/Users/Subham/llmteam/rava-llm-team-workflows/Fine tuning dataset - Sheet1.csv")
    qa_pair = []
    for i in range(df.shape[0]):
        qa_pair.append(dspy.Example(question=df["Input"][i], answer=df["Output"][i]).with_inputs("question"))
    with st.spinner():
        teleprompter = BootstrapFewShot(metric=validate_hops)
        optimized_program = teleprompter.compile(CoT(), trainset=qa_pair)
        pred = optimized_program(input)
        st.header("Predicted answer:")
        st.write(pred.answer)
        st.info(f"Answer without training: {gpt3_turbo(input)}")

