import os
import torch
import random
import clip
import numpy as np

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm import tqdm
from PIL import Image
from itertools import repeat 

if __name__ == '__main__':

    if not os.path.exists('./outputs'): # make output dir
        os.makedirs('./outputs')

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False) # pretrained model

    # cast fp32 to use vit-b/32 model
    model = model.float()

    # process image-text data
    class ImgTextDataset(Dataset):
        def __init__(self, list_image_path, list_text):
            self.image_path = list_image_path
            self.title  = clip.tokenize(list_text) # tokenize everthing

        def __len__(self):
            return len(self.title)

        def __getitem__(self, idx):
            image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
            title = self.title[idx]
            return image,title
    
    # read text file
    def read_file_lines(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines()]
            return lines
    
    # image-text train_dataset
    entries = os.listdir('./db/cars_train')
    text_path = './db/cars.txt'

    load_text = read_file_lines(text_path)
    list_image_path = [] # 2151*250 = 537642
    list_text= [] # 2151*250 = 537642

    #shape_dict = {} # key : list_text, value : image_paths
    shape_dict_index = {} # key : list_text, value : image_paths(index)

    for count, entry in enumerate(entries):
        image_dir = f'./db/cars_train/{entry}/rgb/'
        image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]

        list_image_path.extend(image_paths)
        list_text.extend(repeat(load_text[count], len(image_paths)))

        key = load_text[count]

        # shape_dict 만들기
        # key가 이미 존재하면 해당 key의 value(리스트)에 이미지 경로들을 추가
        # key가 존재하지 않으면 새로운 key와 빈 리스트를 생성하고 이미지 경로들을 추가
        #shape_dict.setdefault(key, []).extend(image_paths)

        # shape_dict_index 만들기
        # key가 이미 존재하면 해당 key의 value(리스트)에 이미지 인덱스들을 추가
        # key가 존재하지 않으면 새로운 key와 빈 리스트를 생성하고 이미지 인덱스들을 추가
        shape_dict_index.setdefault(key, []).extend(range(len(list_image_path) - len(image_paths), len(list_image_path)))

    shape_dict_index_key = list(shape_dict_index.keys())

    # image-text val_dataset
    val_entries = os.listdir('./db/cars_train_val')
    val_image_path =[] # 2150*10 = 21510
    val_text = [] # 2150*10 = 21510
    val_dict_indexs = {} # key : list_text, value : val_image_paths(index)
    
    for count, entry in enumerate(val_entries) :
        val_image_dir = f'./db/cars_train_val/{entry}/rgb/'
        val_image_paths = [os.path.join(val_image_dir, file) for file in os.listdir(val_image_dir)]
        
        val_image_path.extend(val_image_paths)
        val_text.extend(repeat(load_text[count], len(val_image_paths)))

        key = load_text[count]

        val_dict_indexs.setdefault(key, []).extend(range(len(val_image_path) - len(val_image_paths), len(val_image_path)))
    
    # value shuffled
    val_dict_index = {}
    for key, values in val_dict_indexs.items():
        val_values = values.copy()  
        random.shuffle(val_values) 
        val_dict_index[key] = val_values

    # custom random sampler
    class MyBatchSampler(torch.utils.data.Sampler):
        
        def __init__(
            self,
            text_superset,
            text2image_indices_dict,
            num_batches: int,
            batch_size: int,
        ):
            super().__init__(None)
            """
            Args
                text_superset: ex) ['a red car', ..., 'a blue car']
                text2image_indices_dict: dictionary (key: text, value: indices)
                    ex)
                        key: text
                        value: [0, 5, 80, 5000, 14068]
                        즉, value는 2151*250 개의 image path들의 indices
                num_batches: batch의 개수 제한
                batch_size: batch 크기
            """
            self.text_superset = text_superset
            self.text2image_indices_dict = text2image_indices_dict
            self.num_batches = num_batches
            self.batch_size = batch_size

        def __iter__(self):
            
            batch_counter = 0
            while batch_counter < self.num_batches:
                # ['a red car', ..., 'a blue car']로부터 subset 추출
                # sub_text_indices = [ 3, 0, 5, 1, 4 ...]
                sub_text_indices = torch.randperm(len(self.text_superset))[
                    : self.batch_size
                ]
                # text_subset에 text_superset 인덱스에 해당하는 텍스트들을 저장 
                text_subset = [
                    self.text_superset[sub_text_indices[i]] for i in range(self.batch_size)
                ]
                # text_subset의 각 text에 해당하는 image_path들 중 random하게 1씩 추출
                # 각 image_path는 전체 image_pathes의 어떤 index에 해당함
                batch = []
                for text in text_subset:
                    _image_indices = self.text2image_indices_dict[text]
                    batch.append(random.choice(_image_indices))
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

        def __len__(self):
            return self.num_batches * self.batch_size
    
    # val sampler
    class ValBatchSampler(torch.utils.data.Sampler):
        
        def __init__(
            self,
            text_superset,
            text2image_indices_dict,
            num_batches: int,
            batch_size: int,
        ):
            super().__init__(None)
            """
            Args
                text_superset: ex) ['a red car', ..., 'a blue car']
                text2image_indices_dict: dictionary (key: text, value: indices)
                    ex)
                        key: text
                        value: [0, 5, 80, 5000, 14068]
                        즉, value는 2151*250 개의 image path들의 indices
                num_batches: batch의 개수 제한
                batch_size: batch 크기
            """
            self.text_superset = text_superset
            self.text2image_indices_dict = text2image_indices_dict
            self.num_batches = num_batches
            self.batch_size = batch_size

        def __iter__(self):
            
            batch_counter = 0
            while batch_counter < self.num_batches:
                # sub_text_indices = [ 0, 1, 2, 3, 4 ...]
                sub_text_indices = torch.arange(self.batch_size)
                # text_subset에 text_superset 인덱스에 해당하는 텍스트들을 저장 
                text_subset = [self.text_superset[i] for i in sub_text_indices]
                # text_subset의 각 text에 해당하는 image_path들 중 random하게 1씩 추출
                # 각 image_path는 전체 image_pathes의 어떤 index에 해당함
                batch = []
                for text in text_subset:
                    _image_indices = self.text2image_indices_dict[text]
                    batch.append(_image_indices[batch_counter % len(_image_indices)])
                yield batch
                batch_counter += 1

        def __len__(self):
            return self.num_batches * self.batch_size
        
    # set args
    kwargs = {"num_workers": 0, "pin_memory": True} # window = 0
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    num_epoch = 100
    train_batches = 10000 # if batch_size = 43, 12503
    val_batches = 400 # If batch_size = 43, 500
    num_batch_size = 43

    text_superset = shape_dict_index_key
    text2image_indices_dict = shape_dict_index

    # Early stopping
    patience = 4
    best_loss = float('inf')
    counter = 0

    # load train dataset
    train_dataset = ImgTextDataset(list_image_path, list_text)
    my_batch_sampler = MyBatchSampler(text_superset, text2image_indices_dict, train_batches, num_batch_size)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=my_batch_sampler, shuffle=False, **kwargs)
    
    # load val dataset
    val_dataset = ImgTextDataset(val_image_path, val_text)
    val_batch_sampler = ValBatchSampler(text_superset, val_dict_index, val_batches, num_batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler, shuffle=False, **kwargs)    

    # AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6, weight_decay=1e-5) #L2 
    epoch_losses = []
    val_losses = []

    # train and validate process
    for epoch in range(num_epoch):
        
        model.train()
        train_loss = 0.0 # initialize epoch loss

        for batch in tqdm(train_dataloader,total=train_batches, desc=f'epoch : {epoch + 1}/{num_epoch}'):

            # gradient zero
            optimizer.zero_grad()

            # input values
            images,texts = batch

            images= images.to(device)
            texts = texts.to(device)
            
            # expectation values
            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
 
            # back-propagation
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()

            # accumulate batch loss
            train_loss += total_loss

            # optimizer to next step
            optimizer.step()

        epoch_losses.append(train_loss.item() / len(train_dataloader))
           
        # Print the epoch loss
        print(f'Epoch {epoch + 1}/{num_epoch}, Loss: {epoch_losses[-1]}')

        model.eval()
        val_loss = 0.0 # initialize val loss

        with torch.no_grad():
            for batch in tqdm(val_dataloader, total=val_batches, desc=f'val_epoch : {epoch + 1}/{num_epoch}'):

                optimizer.zero_grad()

                images,texts = batch

                images= images.to(device)
                texts = texts.to(device)
                
                logits_per_image, logits_per_text = model(images, texts)

                ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                
                val_total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                val_loss += val_total_loss

            val_losses.append(val_loss.item() / len(val_dataloader))
            
            # print the epoch loss
            print(f'val_Epoch {epoch + 1}/{num_epoch}, val_Loss: {val_losses[-1]}')
            
        if val_loss < best_loss: # Early stopping check
            best_loss = val_loss
            counter = 0
            
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    torch.save(model.state_dict(), os.path.join('./outputs', f'clip_trained.pth'))
