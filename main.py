import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import emoji
import googletrans

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

change = {'anger' : ":face_with_symbols_on_mouth:",
          'disgust' : ":nauseated_face:",
          'fear' : ":fearful_face:",
          'joy' : ":grinning_face:",
          'neutral' : ":neutral_face:",
          'sadness' : ":loudly_crying_face:",
          'surprise' : ":astonished_face:"
          }

translator = googletrans.Translator()

tokenizer = AutoTokenizer.from_pretrained(
    'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
    bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)
model = AutoModelForCausalLM.from_pretrained(
    'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype='auto', low_cpu_mem_usage=True
).to(device='cuda', non_blocking=True)
_ = model.eval()


def gpt_interview(prompt, max_length: int = 256, ends_interview: bool = False):
    with torch.no_grad():
        model.eval
        # 입력문장을 토크나이저를 사용하여 토큰화
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
        # 토큰화된 문장을 입력으로 토큰형태의 새로운 문장 생성
        gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=max_length)
        # 생성된 문장을 다시 문자열 형태로 디코딩
        generated = tokenizer.batch_decode(gen_tokens)[0]
    generated_answer = generated[len(prompt):]
    # if ends_interview:
    #     end_idx = generated_answer.index('\n', 2)
    #     return generated[:len(prompt)+end_idx-3]
    # else:
    end_idx = generated_answer.index('Q')
    return generated[len(prompt):len(prompt) + end_idx - 1]


major = input("어느 학과 교수님과 인터뷰 할까요?\n")
time.sleep(2)
first = input(f'{major} 교수님과 인터뷰를 진행하겠습니다. 질문을 해주세요.\n')

prompt = f"""{major} 교수님과 인터뷰를 진행하겠습니다.
Q:{first}
A:"""

user = ''
while user!="종료":
    chatbot = gpt_interview(prompt, max_length=400)

    for sentence in chatbot.split('.'):
        try:
            ko_en = translator.translate(sentence, dest='en', src='ko')

            emotion = classifier(ko_en.text)
            top_emotion = ""
            top_score = 0
            for num in range(len(emotion[0])):
                if top_score < emotion[0][num]['score']:
                    top_emotion = emotion[0][num]['label']
                    top_score = emotion[0][num]['score']
            print(f"{sentence}{emoji.emojize(change[top_emotion])}")
            time.sleep(1)
        except:
            continue

    user = input('')
    prompt = f"""{major} 교수님과 인터뷰를 진행하겠습니다.
    Q:{major} 교수님 안녕하세요.
    A:네 안녕하세요. {major} 교수입니다.
    Q:질문이 있습니다. {user}
    A:"""
