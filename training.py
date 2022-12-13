from torch.optim import AdamW
from transformers import DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

def data() :
    raw_train = load_dataset('csv', data_files='processing.csv', encoding='cp949')
    dataset = DatasetDict({'train': raw_train})
    tokenizer = AutoTokenizer.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]', max_length=2048,truncation=True, padding=True
    )

    def tokenize_function(example):
        return tokenizer(str(example["question"]),str(example["answer"]), truncation=True, padding=True)


    tokenized_datasets = raw_train.map(tokenize_function, batched=True)
    print(tokenized_datasets)


    tokenized_datasets = tokenized_datasets.remove_columns(["question","Unnamed: 0"])
    tokenized_datasets['train'] = tokenized_datasets['train'].rename_column("answer", "labels")
    tokenized_datasets.set_format("torch")
    # tokenized_datasets = tokenized_datasets.remove_columns(raw_train["train"].column_names)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_datasets["train"],  batch_size=1, collate_fn=data_collator, shuffle=True)

    return train_dataloader

def train():
    raw_train = load_dataset('csv', data_files='processing.csv', encoding='cp949').remove_columns('Unnamed: 0')

    tokenizer = AutoTokenizer.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
    )

    model = AutoModelForCausalLM.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype='auto', low_cpu_mem_usage=True
    ).to(device='cuda', non_blocking=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = AdamW(model.parameters(), lr=1e-6)

    criterion = nn.CrossEntropyLoss()

    top_acc=0
    for epoch in range(1):
        total_acc_train = 0
        total_loss_train = 0
        now_acc_train = 0
        now_loss_train=0

        train_dataloader = tqdm(raw_train['train'], desc='Loading train dataset')
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch1 = tokenizer(batch['question'], batch['answer'],return_tensors='pt', max_length=128, truncation=True, padding=True)
            #batch2 = tokenizer(batch['answer'],return_tensors='pt', max_length=512,truncation=True, padding=True)

            batch1 = {k: v.to(device) for k, v in batch1.items()}
            #batch2 = {k: v.to(device) for k, v in batch2.items()}

            output = model(input_ids = batch1['input_ids'], token_type_ids=batch1['token_type_ids'], attention_mask=batch1['attention_mask'], labels= batch1['input_ids'])

            # loss = criterion(output.logits, batch['labels'])
            output.loss.backward()
            optimizer.step()


            total_loss_train = total_loss_train + output.loss.item()

            now_loss_train +=1


            train_dataloader.set_description(
                "Loss %.04f | step %d Epoch %d" % (total_loss_train / now_loss_train, i, epoch))


        print(f"| epoch : {epoch}"
              f"| train loss : {total_loss_train / now_loss_train}")
    model.save_pretrained('model.pt')


if __name__=="__main__":

    #train_dataloader =data()
    # 학습 코드
    train()
