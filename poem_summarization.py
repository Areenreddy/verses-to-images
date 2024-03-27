import openai
import pandas as pd
import os
import pickle
import csv
import numpy as np
import time
from openai import AsyncOpenAI


client = AsyncOpenAI(api_key="YOUR_API_KEY")
prompt = """Generate the summary of the poem without losing it's essence in 15 words"""
messages = [{"role": "system", "content": prompt}]


def limit_words(text, max_words=1900):
  words = text.split()
  if len(words) <= max_words:
    return text
  else:
    return " ".join(words[:max_words])


df=pd.read_csv("poemsum_train.csv")
a=[]

for i in range(len(df)):
  message = limit_words(df['ctext'][i])
  if message:
    messages.append({"role": "user", "content": message})
    chat = await client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
  reply = chat.choices[0].message.content
  a.append(reply)
  messages.pop()


df["our_summary"]=a
df.to_csv("train_poemsummation.csv")