import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Current device:', device)
model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device=device, non_blocking=True)
_ = model.eval()

def gpt(prompt, max_length: int = 256):
    with torch.no_grad():
        # 입력문장을 토크나이저를 사용하여 토큰화
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=device, non_blocking=True)
        # 토큰화된 문장을 입력으로 토큰형태의 새로운 문장 생성
        gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=max_length)
        # 생성된 문장을 다시 문자열 형태로 디코딩
        generated = tokenizer.batch_decode(gen_tokens)[0]
    return generated

# 기존함수를 조금 수정하여 인터뷰를 위한 함수를 새로 생성하겠습니다.
def gpt_interview(prompt, max_length: int = 256, ends_interview: bool = False):
    with torch.no_grad():
        # 입력문장을 토크나이저를 사용하여 토큰화
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=device, non_blocking=True)
        # 토큰화된 문장을 입력으로 토큰형태의 새로운 문장 생성
        gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=max_length)
        # 생성된 문장을 다시 문자열 형태로 디코딩
        generated = tokenizer.batch_decode(gen_tokens)[0]
    generated_answer = generated[len(prompt)-3:]
    if ends_interview:
        end_idx = generated_answer.index('\n', 2)
        return generated[:len(prompt)+end_idx-3]
    else:
        end_idx = generated_answer.index('Q')
        return generated[:len(prompt)+end_idx-3]

user = input("어느 과 교수님과 인터뷰 할까요?")
prompt = f"""
Q: {user} 교수님 안녕하세요.
A:"""
answer = gpt_interview(prompt)
print(answer)
prompt = prompt+answer

while 1:
    temp = input("")
    prompt = prompt +'\nQ: '+temp+'\nA: '

    chatbot = gpt_interview(prompt)
    print(chatbot)
