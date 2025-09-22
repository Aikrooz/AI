import gradio as gr
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tempfile

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

labels = ["complaint", "praise", "suggestion", "neutral"]


def classify_chinese_batch_with_avg(texts):
    if isinstance(texts, str):
        
        texts = [t.strip() for t in texts.split("\n") if t.strip()]
    
    
    translations = translator(texts, max_length=512)
    english_texts = [t["translation_text"] for t in translations]

  
    results = classifier(english_texts, candidate_labels=labels)

    output = []
    all_scores = []
    rows = []
    for orig, trans, res in zip(texts, english_texts, results):
        scores_dict = dict(zip(res["labels"], res["scores"]))
        all_scores.append([scores_dict[l] for l in labels])
        output.append(f"原文: {orig}\n翻译: {trans}\n预测: {res['labels'][0]} {res['scores'][0]*100:.1f}%\n")
        rows.append({"original": orig, "translated": trans, "predicted": res["labels"][0], **scores_dict})


    all_scores = np.array(all_scores)
    avg_scores = dict(zip(labels, all_scores.mean(axis=0) * 100))

    fig, ax = plt.subplots()
    ax.bar(avg_scores.keys(), avg_scores.values())
    ax.set_title("Average Classification Scores (%)")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)

    # Step 5: Save results to CSV
    df = pd.DataFrame(rows)
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmpfile.name, index=False, encoding="utf-8-sig")

    return "\n\n".join(output), fig, tmpfile.name


demo = gr.Interface(
    fn=classify_chinese_batch_with_avg,
    inputs=gr.Textbox(lines=10, placeholder="输入多条中文反馈，每条换行", label="输入文本"),
    outputs=[
        gr.Textbox(label="逐条结果"),
        gr.Plot(label="平均分布图 (百分比)"),
        gr.File(label="下载分类结果 CSV")
    ],
    title="中文反馈分类 (翻译 + 分类链)",
    description="输入中文客户反馈，模型会先翻译成英文，再分类为: 投诉, 表扬, 建议, 中立，并展示平均结果 (百分比)。"
)

if __name__ == "__main__":
    demo.launch()
