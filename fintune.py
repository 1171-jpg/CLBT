import logging
import time
import pytz
import datetime
from logging.handlers import TimedRotatingFileHandler
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForMultipleChoice, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device="cuda:0"


with open('output.log', 'w'):
    pass

tz = pytz.timezone('America/Los_Angeles')
logging.Formatter.converter = lambda *args: datetime.datetime.now(tz).timetuple()

handler = TimedRotatingFileHandler('output.log', when='midnight', utc=True)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
logger.addHandler(handler)


class MultipleChoiceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, attention_mask, label,id = self.data[idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label,
            'ids':id
        }
    
def getDataLoader(dataset,tokenizer,batch_size = 1):
    processed_data = []
    combined_inputs = []
    labels = []
    ids =[]
    for item in dataset:
        question = item["question"]
        for choice in item["choice_list"]:
            combined_inputs.append(question + " [SEP] " + choice)
        labels.append(item["label"])
        ids.append(item['id'])

    inputs = tokenizer(combined_inputs, padding=True, truncation=True, return_tensors="pt", max_length=256)
    num_choices = len(dataset[0]['choice_list'])

    input_shape = inputs["input_ids"].shape[0] // num_choices
    input_ids = inputs["input_ids"].view(input_shape, num_choices, -1)
    attention_mask = inputs["attention_mask"].view(input_shape, num_choices, -1)

    processed_data = [(input_id, att_mask, label, id) for input_id, att_mask, label, id in zip(input_ids, attention_mask, labels, ids)]


    dataset = MultipleChoiceDataset(processed_data)
    dataloader = DataLoader(dataset, shuffle=True,batch_size=batch_size)

    return dataloader



def train(traindata,model,tokenizer):
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 5. 训练
    model.train().cuda()
    losses = 0
    for epoch in tqdm(range(5)):  
        for index, batch in enumerate(traindata):
            input_ids, attention_mask, labels,ids = [val.to(device) if key != "ids" else val for key, val in batch.items()]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if index%25 == 0:
                logger.info("Loss: {}".format(losses/25))
                losses = 0

    # 6. 保存模型
    model.save_pretrained("./qa_roberta_model")
    tokenizer.save_pretrained("./qa_roberta_model")

def eval(testdata,model):
    model.eval()
    preds = []
    true_labels = []
    ture_ids = []
    for batch in tqdm(testdata):
        input_ids, attention_mask, labels,ids = [val.to(device) if key != "ids" else val for key, val in batch.items()]
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        true_labels.append(labels)
        ture_ids.append(ids)

    summary_result = [(id[0],pred.item(),label.item()) for id,pred,label in zip(ture_ids,preds,true_labels)]
    final_result = getFinalResult(summary_result)

def getFinalResult(answer_list):
    group_number = len(answer_list)/3
    answer_dict = {}
    for answer in answer_list:
        id = answer[0].split('-')[1].split('_')[0]
        ad_type = answer[0].split('_')[1] if '_' in answer[0] else ''
        if id not in answer_dict:
            answer_dict[id] = [0,0,0]

        if answer[1] == answer[2]:
            if ad_type == 'SR':
                answer_dict[id][1] = 1
            elif ad_type == 'CR':
                answer_dict[id][2] = 1
            else:
                answer_dict[id][0] = 1

    ori = 0
    sem = 0
    con = 0
    ori_sem = 0
    ori_sem_con = 0
    for value in answer_dict.values():
        if value[0] == 1:
            ori +=1
        if value[1] == 1:
            sem +=1
        if value[2] == 1:
            con +=1
        if value[0] == 1 and value[1] == 1:
            ori_sem +=1
        if value[0] == 1 and value[1] == 1 and value[2] == 1:
            ori_sem_con +=1

    result_list = [ori/group_number,sem/group_number,con/group_number,ori_sem/group_number,ori_sem_con/group_number,(ori+sem+con)/group_number/3]
    logger.info(result_list)
    return result_list


def main():

    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMultipleChoice.from_pretrained(model_name)

    ### load data
    logger.info("--- Load Data ---")
    train_sp = np.load("../Brain_teaser/paper_data/final_version/semeval_data/SP-train.npy",allow_pickle=True)
    train_wp = np.load("../Brain_teaser/paper_data/final_version/semeval_data/WP-train.npy",allow_pickle=True)
    val_sp = np.load("../Brain_teaser/paper_data/final_version/semeval_data/SP-val.npy",allow_pickle=True)
    val_wp = np.load("../Brain_teaser/paper_data/final_version/semeval_data/WP-val.npy",allow_pickle=True)
    test_sp = np.load("../Brain_teaser/paper_data/final_version/semeval_data/SP-test.npy",allow_pickle=True)
    test_wp = np.load("../Brain_teaser/paper_data/final_version/semeval_data/WP-test.npy",allow_pickle=True)
    TrainDataSP = getDataLoader(train_sp,tokenizer,4)
    TrainDataWP = getDataLoader(train_wp,tokenizer,4)
    ValDataSP = getDataLoader(val_sp,tokenizer)
    ValDataWP = getDataLoader(val_wp,tokenizer)
    TestDataSP = getDataLoader(test_sp,tokenizer)
    TestDataWP = getDataLoader(test_wp,tokenizer)

    ### train
    logger.info("--- Start Training ---")
    train(TrainDataWP,model,tokenizer)
    logger.info("--- Start Evaluation ---")
    eval(ValDataWP,model)   
if __name__ == "__main__":
    main()