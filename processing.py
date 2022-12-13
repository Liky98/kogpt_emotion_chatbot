import pandas as pd

dataset = pd.read_csv("dataset/한국직업능력연구원_커리어넷 학과인터뷰_20210928 (1).csv", encoding='cp949')

question_list =[]
answer_list=[]

for i in range(len(dataset)):
    major = dataset['인터뷰제목'][i]
    for j in range(1,20):
        try:
            if not pd.isna(dataset[f'질문_{j}'][i]):
                question = dataset[f'질문_{j}'][i]
                answer = dataset[f'답변_{j}'][i]

                prompt = f"""{major} 교수님과 인터뷰를 진행하겠습니다.
                Q:{major} 교수님 안녕하세요.
                A:네 안녕하세요. {major} 교수입니다.
                Q:질문이 있습니다. {question}
                A:"""

                question_list.append(prompt)
                answer_list.append(answer)
        except:
            break

df = pd.DataFrame({'question' : question_list,
                   'answer' : answer_list})
df.to_csv("processing.csv", encoding='cp949')


