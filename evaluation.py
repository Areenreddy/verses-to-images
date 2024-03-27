from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from evaluate import load


df = pd.read_csv("/content/train_poemsummation.csv")
df1=pd.read_csv("/content/valid_poemsummation.csv")
df=df+df1

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
rouge_1 = []
rouge_2 = []
rouge_l = []

bleu_1 = []
bleu_2 = []
bleu_3 = []
bleu_4 = []

bscore = []

bertscore = load("bertscore")


for i in range(len(df)):

    src_sent = str(df["text"][i])
    tgt_sent = str(df["our_summary"][i])

    scores = scorer.score(src_sent,tgt_sent)

    rouge_1.append(scores["rouge1"][2])
    rouge_2.append(scores["rouge2"][2])
    rouge_l.append(scores["rougeL"][2])

    bleu_1.append(sentence_bleu([tgt_sent],src_sent,weights=(1, 0, 0, 0)))
    bleu_2.append(sentence_bleu([tgt_sent],src_sent,weights=(0, 1, 0, 0)))
    bleu_3.append(sentence_bleu([tgt_sent],src_sent,weights=(0, 0, 1, 0)))
    bleu_4.append(sentence_bleu([tgt_sent],src_sent,weights=(0, 0, 0, 1)))




print("rouge 1",sum(rouge_1)/len(rouge_1))
print("rouge 2",sum(rouge_2)/len(rouge_2))
print("rouge L",sum(rouge_l)/len(rouge_l))


print("bleu 1",sum(bleu_1)/len(bleu_1))
print("bleu 2",sum(bleu_2)/len(bleu_2))
print("bleu 3",sum(bleu_3)/len(bleu_3))
print("bleu 4",sum(bleu_4)/len(bleu_4))





for i in range(len(df)):

    predictions = [df["text"][i]]
    references = [df["our_summary"][i]]
    k=bertscore.compute(predictions=predictions, references=references, lang="en")["f1"][0]
    bscore.append(k)

print("bert score",sum(bscore)/len(bscore))