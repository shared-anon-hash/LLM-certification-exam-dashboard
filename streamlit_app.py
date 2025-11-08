import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

st.title("Language Model Performance Dashboard")

# -------------------------
# Certification Exam Dataset
# -------------------------
st.header("Certification Exam Results")

exam_csv = """
Model,CTFL - A,CTFL - B,CTFL - C,CTFL - D,CTAL-TAv4.1,CTAL-TAv3.1,CTAL-TAEv2.0,CTAL-TAEv1.3,CTAL-TMv3.0,CTAL-TMv1.4,CTAL-ATT,CTAL-TTA,CT-ATLaS,CT-AI,CTFL-AT,CT-GenAI,CT-MAT,CT-TAS,CT-MBT,CT-AcT,CT-PT,CT-SEC,CT-STE,CT-UT,CT-AuT,CT-GaMe,CT-GT,CTEL-ITP-ATP,CTEL-ITP-ITPI,CTEL-TM-OTM,Total Passed
gpt-5,37.5p,35p,37p,35p,70.5p,64.5p,57p,67p,72p,77p,22.5p,71.5p,61p,43p,37.5p,44p,35p,42.5p,40p,37p,33p,55p,34.5p,34p,37p,27p,26p,28p,34p,41p,30
gpt-5-mini,36.5p,36.5p,32p,37p,59.5p,59p,52.5p,58p,68p,71p,21.5p,71.5p,60p,40p,37p,43p,35p,41.5p,38p,35p,29p,57p,32.5p,31p,34p,28p,21f,30p,35p,41.3p,29
gpt-5-nano,32p,31p,31p,36p,59.5p,50.5f,50.5p,55.5p,60p,69p,18.5f,61.5p,56p,37p,31.5p,38p,31p,31.5f,34p,33p,30p,62p,31p,30p,30p,28p,21f,26p,28p,29.1f,25
gpt-5-2025-08-07,38.5p,35p,37p,37p,77.5p,68.5p,57p,65p,73p,72p,22.5p,72.5p,61p,41p,36.5p,42p,33p,43.5p,40p,37p,30p,66p,34p,36p,37p,26p,27p,30p,36p,45p,30
gpt-5-mini-2025-08-07,36.5p,36p,33p,36p,61.5p,65p,56.5p,58p,67p,68.5p,24.5p,71p,62p,37p,36p,43p,36p,42.5p,38p,33p,31p,63p,35.5p,32p,35p,27p,25f,27p,35p,41.3p,29
gpt-5-nano-2025-08-07,32p,31p,31p,36p,59.5p,46.5f,45p,63p,56f,62.5p,22.5p,60p,58p,37p,32.5p,42.5p,28p,34.5p,33.5p,33p,29p,59p,31p,32p,30p,24f,24f,27p,27p,29.1f,25
gpt-5-chat-latest,34.5p,29p,32p,30.5p,51.5p,49f,48p,65p,64p,71.5p,20.5p,59.5p,62p,36p,34.5p,44p,33p,37p,36p,33p,28p,58p,32p,35p,37p,28p,23f,26p,34p,37.3p,28
gpt-4.1,34.5p,29.5p,28p,33.5p,46.5f,52.5p,51p,62p,67p,72.5p,20p,60p,62p,38p,32.5p,45p,31p,35.5p,38p,34p,30p,57p,31.5p,37p,35p,28p,25f,25p,34p,37.3p,28
gpt-4.1-mini,32p,26p,27p,27p,45.5f,42f,54p,60p,66.5p,69p,19.5p,56p,61p,37p,33p,42p,31p,37p,36.5p,31p,32p,60p,34.5p,34p,34p,28p,26p,26p,34p,34.3p,28
gpt-4.1-nano,31p,22f,26p,28.5p,37f,35.5f,44f,52.5p,58.5p,59.5p,21.5p,43.5f,54.5p,29f,29.5p,35.5p,31p,27.5f,27p,29p,27p,50f,27.5f,28p,28p,26.5p,21f,19f,19f,20.1f,17
gpt-4.1-2025-04-14,33.5p,30.5p,29p,32.5p,41f,46.5f,50p,62p,65p,73p,21p,61p,62p,38p,33.5p,45p,33p,35.5p,38p,34p,30p,55p,31.5p,35p,35p,26p,30p,26p,33p,37.6p,28
gpt-4.1-mini-2025-04-14,30p,27p,30p,27p,45.5f,34f,52p,58p,63.5p,67.5p,22.5p,57p,60p,38p,32p,43p,31p,38p,35.5p,30p,29p,57p,34p,32p,37p,26.5p,24f,25p,31p,36.6p,27
gpt-4.1-nano-2025-04-14,29p,21f,23f,31.5p,37f,39f,47p,54.5p,59.5p,60.5p,18.5f,41f,55.5p,33p,30p,37.5p,32p,24f,27p,24f,25f,45f,27f,26p,26p,23.5f,22f,18f,23f,24.6f,14
o4-mini,36.5p,33.5p,34p,36p,56.5p,59.5p,55.5p,57.5p,62p,69p,22.5p,67p,61p,38p,37p,44p,36p,40p,39p,35p,30p,68p,36p,34p,34p,25f,22f,29p,30p,43.3p,28
o4-mini-2025-04-16,35.5p,33.5p,34p,36p,73.5p,67p,49.5p,58.5p,60p,72.5p,22.5p,66p,60p,40p,37.5p,44p,36p,38p,39p,33p,35p,62p,35p,33p,38p,27p,24f,32p,34p,37.3p,29
o3,35.5p,37p,36p,38p,76.5p,70.5p,56p,68p,73p,75p,22.5p,71.5p,61p,41p,35.5p,44p,36p,41p,39p,34p,29p,60p,34p,35p,37p,27p,23f,26p,34p,39.3p,29
o3-2025-04-16,38.5p,37p,36p,36p,71.5p,69.5p,55p,64p,66p,74p,25.5p,71.5p,60p,40p,35p,45p,36p,40p,38.5p,33p,31p,64p,33.5p,35p,36p,28.5p,25f,27p,36p,40.3p,29
o1,38.5p,36p,34p,35p,62.5p,62.5p,52p,67p,65p,74.5p,22.5p,68.5p,60p,40p,36p,40.5p,35p,36p,40p,31p,33p,58p,31p,32p,35p,26.5p,24f,27p,35p,44.3p,29
o1-2024-12-17,34.5p,34p,34p,37p,66.5p,61.5p,54p,67p,66p,74.5p,19.5p,68p,59p,40p,36p,41.5p,34p,39p,40p,30p,33p,56p,34p,33p,35p,26.5p,22f,28p,33p,37.3p,29
gpt-4o,36.5p,29.5p,29p,29.5p,39f,39.5f,55.5p,62p,74p,71p,20.5p,61.5p,59.5p,37p,34.5p,41p,33p,32p,36.5p,34p,30p,57p,32.5p,34p,32p,29p,22f,27p,34p,31.3p,27
gpt-4o-2024-11-20,32.5p,27.5p,30p,31p,48f,44.5f,54.5p,64p,67p,71p,24.5p,59.5p,59.5p,38p,35.5p,39.5p,30p,37p,36.5p,34p,29p,53p,32.5p,35p,36p,27p,21f,26p,30p,31.3p,27
gpt-4o-2024-08-06,33.5p,30p,29p,29.5p,39f,46.5f,52p,64p,64p,70p,20.5p,62p,60p,36p,34.5p,41.5p,31p,33p,35.5p,34p,31p,62p,32.5p,35p,31p,28.5p,22f,23f,34p,32.3p,26
gpt-4o-2024-05-13,33.5p,28.5p,31p,31p,48.5f,46.5f,55p,62p,69p,68p,23.5p,60.5p,61p,35p,33p,38.5p,31p,34.5p,35.5p,34p,31p,57p,32p,35p,35p,25.5f,21f,24p,28p,28.3f,25
chatgpt-4o-latest,33.5p,26.5p,29p,29p,49f,55p,51p,65p,68p,68.5p,21.5p,67p,57.5p,37p,34.5p,45p,35p,34p,36p,32p,32p,58p,31.5p,36p,38p,28p,25f,25p,33p,32.3p,28
gpt-4o-mini-2024-07-18,32.5p,22f,27p,29p,40.5f,41.5f,45.5p,55.5p,65p,60.5p,15.5f,39f,55p,31p,34.5p,41p,27.5p,38.5p,31.5p,27p,27p,45f,30.5p,29p,29p,27.5p,23f,20f,22f,27.8f,20
gpt-4-turbo,33.5p,26p,28p,30.5p,44f,42f,51p,56p,61p,66.5p,23.5p,45.5f,56.5p,35p,33.5p,40.5p,32p,36p,36.5p,32p,30p,52p,33p,31p,32p,26p,23f,26p,26p,30.5p,26
gpt-4-turbo-2024-04-09,32.5p,27p,29p,29.5p,44.5f,41f,52p,57p,57f,65p,23.5p,48.5f,56.5p,33p,35.5p,38.5p,33p,36p,35.5p,32p,32p,52p,31p,32p,35p,26p,22f,23f,30p,36.6p,24
gemini-2.5-pro,39.5p,36p,33p,36p,66.5p,67p,60p,64p,70p,76p,27.5p,63p,59p,42p,37p,44p,35p,41p,39p,36p,33p,69p,35p,33p,36p,31p,25f,27p,34p,39p,29
gemini-2.5-flash,36.5p,33p,30p,37p,42f,47.5f,51p,61p,64p,68.5p,20.5p,56.5p,59p,40p,33p,42.5p,33p,35.5p,37p,33p,31p,57p,32.5p,32p,35p,27p,27p,22f,31p,32.3p,27
gemini-2.5-flash-lite,35p,24.5f,25f,29p,38.5f,35.5f,55p,59p,72p,70p,21.5p,47f,55p,34p,34p,45p,33p,35.5p,33p,32p,29p,50f,32.5p,30p,31p,24f,14f,22f,29p,33.1p,21
gemini-2.5-flash-lite-preview-09-2025,34p,30.5p,28p,30.5p,35.5f,42.5f,50p,61p,65p,71.5p,22p,45.5f,51.5p,38p,31p,43.5p,31p,35p,34.5p,30p,31p,64p,33.5p,31p,31p,26p,17f,23f,32p,29.7f,24
gemini-2.0-flash,33p,29p,28p,28p,48.5f,43f,55.5p,56p,74p,71p,24.5p,47f,54p,39p,31.5p,42.5p,32p,40.5p,36.5p,33p,32p,66p,31p,32p,30p,26p,24f,22f,22f,35p,24
gemini-2.0-flash-lite,32p,27p,27p,28.5p,44.5f,32.5f,48.5p,56p,60p,72.5p,17.5f,47.5f,54p,32p,33p,44p,34p,34p,34.5p,33p,29p,58p,34.5p,32p,31p,26p,20f,28p,27p,40.3p,25
qwen2.5vl:32b,27p,23.5f,26.5p,26p,46f,44.5f,44.5f,54p,59p,65.5p,16f,54p,51p,32p,34p,37p,27.5p,38.5p,32.5p,29p,32p,49f,34.5p,31p,31p,24f,26p,24p,26p,34p,23
gemma3:27b,26.5p,25.5f,26p,28.5p,33.5f,39.5f,52.5p,59.5p,62.5p,66.5p,18.5f,52p,56p,33p,33p,41p,30p,29f,31.5p,30p,32p,55p,33.5p,31p,34p,26p,25f,28p,26p,31.3p,24
llava:34b,20f,16f,23.5f,18f,35f,26f,42.5f,52.5p,49f,58p,15f,37.5f,39f,22f,29.5p,27f,25f,32.5p,27p,26p,25f,47f,29.5p,15f,23f,15.5f,17f,17f,18f,19.4f,7
llava:7b,17.5f,17.5f,11.5f,12.5f,30f,31f,29.5f,31.5f,38.5f,46f,11f,38f,28f,20f,16.5f,22f,19.5f,23f,19f,13f,21f,23f,22f,13f,16f,10.5f,17f,23f,18f,21.3f,0
bakllava:latest,15.5f,13f,15f,12f,17f,15.5f,25f,33f,29.5f,32.5f,11f,27f,20.5f,17f,12.5f,24f,12f,19.5f,15f,16f,26p,33f,15f,19f,18f,10.5f,13f,12f,13f,9.9f,1
llava-phi3:latest,19f,10f,21.5f,13f,23f,33f,27.5f,48.5f,43f,41f,17.5f,35f,30f,22f,20.5f,21f,25.5f,25.5f,21f,15f,24f,26f,22f,16f,23f,18.5f,15f,18f,15f,18.5f,0
minicpm-v:latest,18.5f,18f,20f,17.5f,17.5f,28f,38f,37f,40.5f,46f,13f,28.5f,42f,16f,25.5f,21.5f,20f,23.5f,23f,18f,18f,39f,28.5p,17f,19f,14.5f,13f,14f,17f,27.1f,1
llava-llama3:latest,17f,14.5f,15f,15.5f,19f,35.5f,28.5f,26.5f,34.5f,38f,11f,30.5f,28.5f,18f,16f,18f,16.5f,17.5f,9.5f,17f,20f,32f,23f,19f,19f,18f,19f,15f,23f,10.2f,0
mistral-small3.2,27p,27.5p,22.5f,26.5p,35.5f,31f,45p,58p,66p,55.5p,20.5p,46f,50.5p,30f,31p,39.5p,31p,36.5p,33p,29p,28p,58p,32p,34p,26p,25f,22f,25p,24f,25.6f,21
qwen2.5vl_latest,25f,22f,20f,21.5f,27.5f,41f,36f,43f,47.5f,53.5f,21.5p,32.5f,36f,24f,27p,30.5p,26.5p,25.5f,25f,27p,21f,42f,32p,25f,25f,21f,17f,14f,20f,21.7f,6
qwen2.5vl:3b,20.5f,17.5f,21f,16.5f,29.5f,33f,32.5f,45.5f,45.5f,37f,11f,33.5f,33.5f,22f,22.5f,20.5f,21f,20.5f,20.5f,20f,25f,39f,26f,24f,19f,18.5f,20f,12f,22f,19f,0
qwen2.5vl:7b,26p,21.5f,22f,21.5f,38.5f,36.5f,34.5f,45f,56f,47f,21.5p,35f,50.5p,30f,27.5p,33.5p,24.5f,27f,29.5p,20f,22f,43f,27f,24f,26p,24f,14f,20f,25p,27.1f,8
mistral-small3.1:latest,30.5p,28.5p,22.5f,24.5f,35.5f,35f,49.5p,53.5p,58.5p,64p,22.5p,42.5f,50p,29f,28p,41.5p,29p,33p,31.5p,29p,31p,45f,31p,26p,29p,20f,16f,27p,23f,21.3f,19
gemma3:4b,23.5f,12.5f,18f,17.5f,33.5f,27.5f,35f,44f,49.5f,46f,16.5f,29.5f,44.5f,25f,27.5p,32.5p,27.5p,29.5f,28p,25f,24f,41f,27.5f,19f,23f,22f,16f,22f,14f,21.7f,4
gemma3:12b,30p,25.5f,23.5f,25.5f,31.5f,35.5f,45p,55.5p,63p,53.5f,17.5f,49f,52p,30f,31.5p,34.5p,32p,32.5p,34p,25f,28p,56p,29.5p,29p,32p,26.5p,23f,23f,21f,27f,16
granite3.2-vision:2b,19f,10.5f,14.5f,14f,21.5f,32f,26.5f,39.5f,48.5f,31.5f,11.5f,24f,27f,18f,16.5f,17.5f,15f,25.5f,21.5f,13f,14f,30f,20f,9f,20f,15.5f,18f,13f,14f,22.1f,0
random,13.5f,10.5f,9f,10.5f,23f,35f,19f,23f,21f,29f,7f,23.5f,18.5f,12f,8f,10f,12f,7f,12.5f,11f,13f,24f,13.5f,14f,9f,9f,9f,5f,9f,13.4f,0
"""  
df_exam = pd.read_csv(StringIO(exam_csv))

# Convert p/f to numeric
def convert_score(score):
    if isinstance(score, str):
        if 'p' in score:
            return float(score.replace('p',''))
        elif 'f' in score:
            return float(score.replace('f',''))
    return score

score_cols = df_exam.columns[1:]
df_exam[score_cols] = df_exam[score_cols].applymap(convert_score)

# Highlight best per exam
def highlight_best(row):
    max_val = row.max()
    return ['background-color: lightgreen' if v == max_val else '' for v in row]

# Model selection
models_selected_exam = st.multiselect(
    "Select Model(s) to Compare (Exam Results)", 
    df_exam["Model"].unique(), 
    default=df_exam["Model"].iloc[0]
)

if models_selected_exam:
    st.subheader("Exam Scores Table with Highlighted Best per Exam")
    st.dataframe(df_exam[df_exam["Model"].isin(models_selected_exam)].style.apply(highlight_best, subset=score_cols))
    
    st.subheader("Exam Scores Comparison")
    fig, ax = plt.subplots(figsize=(16,6))
    for model in models_selected_exam:
        model_data = df_exam[df_exam["Model"] == model]
        ax.plot(score_cols, model_data[score_cols].iloc[0], marker='o', label=model)
    ax.set_xticklabels(score_cols, rotation=90)
    ax.set_ylabel("Score")
    ax.set_title("Certification Exam Scores Comparison")
    ax.legend()
    st.pyplot(fig)

# -------------------------
# Key Metrics Dataset
# -------------------------
st.header("Key Metrics Dataset")

metrics_csv = """
Model Name,K1,K2,K3,K4,Total
gpt-5,110 , 579 , 203 , 109,1001
 gpt-5-mini,106 , 556 , 190 , 101,953
 gpt-5-nano,95 , 515 , 169 , 91,870
 gpt-5-2025-08-07,109 , 583 , 208 , 111,1011
 gpt-5-mini-2025-08-07,104 , 557 , 198 , 105,964
 gpt-5-nano-2025-08-07,93 , 505 , 168 , 92,858
 gpt-5-chat-latest,104 , 548 , 164 , 94,910
 gpt-4.1,102 , 553 , 166 , 96,917
 gpt-4.1-mini,100 , 532 , 155 , 91,878
 gpt-4.1-nano,83 , 449 , 126 , 76,734
 gpt-4.1-2025-04-14,104 , 556 , 161 , 94,915
 gpt-4.1-mini-2025-04-14,97 , 527 , 150 , 90,864
 gpt-4.1-nano-2025-04-14,82 , 443 , 133 , 73,731
 o4-mini,105 , 550 , 192 , 102,949
 o4-mini-2025-04-16,106 , 554 , 199 , 106,965
 o3,105 , 576 , 205 , 109,995
 o3-2025-04-16,104 , 576 , 203 , 108,991
 o1,105 , 553 , 189 , 107,954
 o1-2024-12-17,102 , 550 , 194 , 104,950
 gpt-4o,102 , 533 , 158 , 95,888
 gpt-4o-2024-11-20,100 , 530 , 163 , 95,888
 gpt-4o-2024-08-06,98 , 527 , 158 , 96,879
 gpt-4o-2024-05-13,97 , 529 , 157 , 96,879
 chatgpt-4o-latest,104 , 544 , 160 , 100,908
 gpt-4o-mini-2024-07-18,90 , 488 , 139 , 71,788
 gpt-4-turbo,97 , 510 , 153 , 87,847
 gpt-4-turbo-2024-04-09,97 , 515 , 152 , 87,851
 gemini-2.5-pro,107 , 583 , 201 , 107,998
 gemini-2.5-flash,104 , 540 , 166 , 93,903
 gemini-2.5-flash-lite,102 , 494 , 148 , 85,829
 gemini-2.5-flash-lite-preview-09-2025,98 , 499 , 148 , 91,836
 gemini-2.0-flash,100 , 535 , 152 , 90,877
 gemini-2.0-flash-lite,99 , 513 , 144 , 85,841
 qwen2.5vl:32b,94 , 480 , 149 , 83,806
 gemma3:27b,97 , 495 , 136 , 90,818
 llava:34b,69 , 388 , 112 , 55,624
 llava:7b,45 , 289 , 78 , 49,461
 bakllava:latest,52 , 258 , 61 , 35,406
 llava-phi3:latest,55 , 336 , 72 , 54,517
 minicpm-v:latest,61 , 327 , 80 , 54,522
 llava-llama3:latest,60 , 267 , 79 , 45,451
 mistral-small3.2,92 , 471 , 139 , 78,780
 qwen2.5vl_latest,76 , 403 , 103 , 59,641
 qwen2.5vl:3b,67 , 340 , 88 , 52,547
 qwen2.5vl:7b,78 , 417 , 110 , 62,667
 mistral-small3.1:latest,91 , 461 , 136 , 70,758
 gemma3:4b,72 , 379 , 102 , 56,609
 gemma3:12b,89 , 463 , 133 , 72,757
 granite3.2-vision:2b,53 , 259 , 78 , 47,437
 random,25 , 151 , 66 , 41,283
"""  # Add all remaining rows for K1-K4 totals

df_metrics = pd.read_csv(StringIO(metrics_csv))

# Model selection
models_selected_metrics = st.multiselect(
    "Select Model(s) to Compare (Key Metrics)", 
    df_metrics["Model Name"].unique(),
    default=df_metrics["Model Name"].iloc[0]
)

if models_selected_metrics:
    st.subheader("Key Metrics Table")
    st.dataframe(df_metrics[df_metrics["Model Name"].isin(models_selected_metrics)])
    
    st.subheader("Key Metrics Comparison")
    fig2, ax2 = plt.subplots(figsize=(12,6))
    for model in models_selected_metrics:
        model_data = df_metrics[df_metrics["Model Name"] == model]
        ax2.plot(['K1','K2','K3','K4'], model_data[['K1','K2','K3','K4']].iloc[0], marker='o', label=model)
    ax2.set_ylabel("Value")
    ax2.set_title("Key Metrics Comparison")
    ax2.legend()
    st.pyplot(fig2)
