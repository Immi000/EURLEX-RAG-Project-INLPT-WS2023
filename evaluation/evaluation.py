from chatmodel import RAG
import numpy as np
import pandas as pd
import git
import time

repo = git.Repo(search_parent_directories=True)


EVALUATION_NAME = "new_embedding"

text_file = open("text_questions_reduced.txt", "r")
lines = text_file.readlines()

QA = RAG(verbose=False)

questions = []
sample_answers = []
model_answers = []
contexts = []

with open(EVALUATION_NAME + "_description.txt", "w") as f:
    f.write(
        f"""
Index name: {QA.index_name}
Retrieved context parts: {QA.context_enrichment}
Number neighboring chunks {QA.use_results}
"""
    )

sha = repo.head.object.hexsha
with open(EVALUATION_NAME + "_description.txt", "a") as f:
    f.write(f"Commit SHA: {sha}\n")

try:
    df = pd.read_pickle(EVALUATION_NAME + "results.pkl")
except:
    df = pd.DataFrame(
        {"questions": [], "sample_answers": [], "model_answers": [], "contexts": []}
    )

i = len(df)
lines = lines[i:]
for line in lines:
    if i > 8:
        break
    QA.clear_history()
    i = i + 1
    split = line.split("=")
    question = split[0].strip()
    desired_answer = split[1].strip()
    output = QA.ask_question(question).strip()
    context = QA.get_context()

    questions.append(question)
    sample_answers.append(desired_answer)
    contexts.append(context)
    model_answers.append(output)

    print("question=", question)
    print("desired_answer=", desired_answer)
    print("model_answer=", output)
    print("-----")

    # df.loc[i] = [question, desired_answer, output, context]

    if i % 1 == 0:
        df = pd.DataFrame(
            {
                "questions": questions,
                "sample_answers": sample_answers,
                "model_answers": model_answers,
                "contexts": contexts,
            }
        )
        df.to_pickle(EVALUATION_NAME + "results.pkl")
    time.sleep(5)
df.to_pickle(EVALUATION_NAME + "results.pkl")
