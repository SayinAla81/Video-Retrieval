import os
import json
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import defaultdict
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor, CLIPModel, TrainingArguments

# تنظیمات اولیه
DATA_FILE = 'dataset.csv'
TEXT_MODEL = 'm3hrdadfi/roberta-zwnj-wnli-mean-tokens'
IMAGE_MODEL = 'openai/clip-vit-base-patch32'
BATCH_SIZE = 32
IMAGE_SIZE = 224
MAX_LEN = 80

# بارگذاری مدل‌ها
print("🔹 در حال بارگذاری مدل‌ها ...")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)  
vision_encoder = AutoModel.from_pretrained(IMAGE_MODEL).eval()
print("✅ مدل‌ها با موفقیت بارگذاری شدند!\n")

# بررسی وجود دایرکتوری
output_dir = 'output_images'
if not os.path.exists(output_dir):
    print(f"Directory '{output_dir}' does not exist. Please check the path.")
else:
    print(f"Directory '{output_dir}' exists.")

# بارگذاری تصاویر
image_paths = [os.path.join(output_dir, fname) for fname in os.listdir(output_dir) if fname.endswith('.png')]
print("Image paths:", image_paths)  # چاپ مسیرهای تصاویر

if __name__ == '__main__':
    from transformers import TrainingArguments, AutoTokenizer, CLIPFeatureExtractor
    vision_preprocessor = CLIPFeatureExtractor.from_pretrained(IMAGE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    args = TrainingArguments(
        "clip-fa",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-5,
        weight_decay=0.003,
        warmup_steps=100,
        fp16=False,
        prediction_loss_only=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        report_to='none'
    )

class VisionDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.preprocess(image), self.image_paths[idx]

    def __len__(self):
        return len(self.image_paths)

class TextDataset(Dataset):
    def __init__(self, text: list, tokenizer, max_len):
        self.len = len(text)
        self.tokens = []
        for t in text:
            encoding = tokenizer.encode_plus(
                t,
                add_special_tokens=True,  
                max_length=max_len,
                padding='max_length',   
                truncation=True,         
                return_tensors='pt'      
            )
            self.tokens.append(encoding)

    def __getitem__(self, idx):
        token = {key: self.tokens[idx][key].squeeze(0) for key in self.tokens[idx].keys()}
        return token

    def __len__(self):
        return self.len
class CLIPDemo:
    def __init__(self, vision_encoder, text_encoder, tokenizer,
                 batch_size: int = 32, max_len: int = 32, device='cuda'):
        self.vision_encoder = vision_encoder.eval().to(device)
        self.text_encoder = text_encoder.eval().to(device)
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_embeddings_ = None
        self.image_embeddings_ = None


    def compute_image_embeddings(self, image_paths):
      print("Starting to compute image embeddings...")
      
      self.image_paths = image_paths
      dataloader = DataLoader(VisionDataset(image_paths=image_paths), batch_size=self.batch_size)
      
      embeddings = []
      print(f"Total number of images: {len(image_paths)}")
      
      with torch.no_grad():
          for batch_idx, (image_tensors, _) in enumerate(dataloader):
              print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
              image_tensors = image_tensors.to(self.device)
              image_embedding = self.vision_encoder(pixel_values=image_tensors).pooler_output
              
              embeddings.append(image_embedding)

      self.image_embeddings_ = torch.cat(embeddings)
      print("Image embeddings computed successfully.")
      print(f"Total embeddings shape: {self.image_embeddings_.shape}")
      return self.image_embeddings_  # اضافه کردن این خط


    def compute_text_embeddings(self, text: list):
        self.text = text
        dataloader = DataLoader(TextDataset(text=text, tokenizer=self.tokenizer, max_len=self.max_len),
                                batch_size=self.batch_size, collate_fn=default_data_collator)
        embeddings = []
        with torch.no_grad():
            for tokens in tqdm(dataloader, desc='computing text embeddings'):
                image_embedding = self.text_encoder(input_ids=tokens["input_ids"].to(self.device),
                                                    attention_mask=tokens["attention_mask"].to(self.device)).pooler_output
                embeddings.append(image_embedding)
        self.text_embeddings_ = torch.cat(embeddings)

    def text_query_embedding(self, query: str = 'موز'):
        tokens = self.tokenizer(query, return_tensors='pt')
        with torch.no_grad():
            text_embedding = self.text_encoder(input_ids=tokens["input_ids"].to(self.device),
                                               attention_mask=tokens["attention_mask"].to(self.device)).pooler_output
        return text_embedding

    def image_query_embedding(self, image):
        image = VisionDataset.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_embedding = self.vision_encoder(
                image.to(self.device)).pooler_output
        return image_embedding

    def most_similars(self, embeddings_1, embeddings_2):
        values, indices = torch.cosine_similarity(
            embeddings_1, embeddings_2).sort(descending=True)
        return values.cpu(), indices.cpu()

    def zero_shot(self, image_path: str):
        image = Image.open(image_path)
        image_embedding = self.image_query_embedding(image)
        values, indices = self.most_similars(image_embedding, self.text_embeddings_)
        for i, sim in zip(indices, torch.softmax(values, dim=0)):
            print(
                f'label: {self.text[i]} | {round(float(sim), 3)}')
        plt.imshow(image)
        plt.axis('off')
    def generate_caption(image_embeddings):
        print("🔹 در حال تولید کپشن برای تصاویر ...")
        captions = {}
        
        if not isinstance(image_embeddings, dict):
            print("⚠️ image_embeddings باید یک دیکشنری باشد!")
            return {}

        for path, img_emb in image_embeddings.items():
            
            best_match = "تصویری از ..."  
            
            captions[path] = best_match
            print(f"📝 کپشن برای {path}: {best_match}")

        print("✅ تولید کپشن‌ها تکمیل شد!\n")
        return captions
    def image_search(query, image_embeddings, top_k=2):
      # محاسبه embedding متن
      query_tokens = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
      query_embedding = text_encoder(**query_tokens).pooler_output[0].detach()

      # دریافت شباهت‌ها و انتخاب بهترین‌ها
      sorted_images = generate_caption(image_embeddings, query_embedding)
      
      # گروه‌بندی تصاویر بر اساس ویدیو
      video_groups = group_images_by_video([img[0] for img in sorted_images])

      # انتخاب دو تصویر از هر ویدیو
      selected_images = []
      for video, images in video_groups.items():
          # برای هر ویدیو دو تصویر با بیشترین شباهت انتخاب می‌شود
          top_images = images[:2]  # دو تصویر اول با بیشترین شباهت
          selected_images.extend(top_images)

      return selected_images

    def analogy(self, input_image_path: str, additional_text: str = 'برف', input_include=True):
        
        base_image = Image.open(input_image_path)
        image_embedding = self.image_query_embedding(base_image)
        additional_embedding = self.text_query_embedding(query=additional_text)
        new_image_embedding = image_embedding + additional_embedding
        _, indices = self.most_similars(
            self.image_embeddings_, new_image_embedding)

        new_image = Image.open(self.image_paths[indices[1 if input_include else 0]])
        _, ax = plt.subplots(1, 2, dpi=100)
        ax[0].imshow(base_image.resize((250, 250)))
        ax[0].set_title('original image')
        ax[0].axis('off')
        ax[1].imshow(new_image.resize((250, 250)))
        ax[1].set_title('new image')
        ax[1].axis('off')
tokenizer = AutoTokenizer.from_pretrained('m3hrdadfi/roberta-zwnj-wnli-mean-tokens')  
vision_encoder = AutoModel.from_pretrained('Model/vision_encoder')  
text_encoder = AutoModel.from_pretrained('Model/text_encoder')  

text_model_path = "Model/text_encoder"
vision_model_path = "Model/vision_encoder"
print("🔹 در حال بارگذاری مدل‌ها ...")

tokenizer = AutoTokenizer.from_pretrained('m3hrdadfi/roberta-zwnj-wnli-mean-tokens')
text_encoder = AutoModel.from_pretrained(text_model_path).eval()
vision_encoder = AutoModel.from_pretrained(vision_model_path).eval()
print("✅ مدل‌ها با موفقیت بارگذاری شدند!\n")

IMAGE_FOLDER = "output_images"

search = CLIPDemo(vision_encoder, text_encoder, tokenizer, device='cpu')
# image_paths = [os.path.join('output_images', fname) for fname in os.listdir('output_images') if fname.endswith('.png')]
image_embeddings = search.compute_image_embeddings(image_paths)
def group_images_by_video(image_paths):
    video_groups = defaultdict(list)
    for path in image_paths:
        video_name = path.split('_')[0]  
        video_groups[video_name].append(path)
    return video_groups

def load_captions_from_file(filename="captions.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
loaded_captions = load_captions_from_file()
search.compute_text_embeddings(list(loaded_captions.values()))

def group_images_by_video(image_paths):
    video_groups = defaultdict(list)
    for path in image_paths:
        video_name = path.split('_')[0]  
        video_groups[video_name].append(path)
    return video_groups

class ImageSearchView(View):
    def get(self, request):
        query = request.GET.get('query', '')
        if query:
            selected_images = self.image_search(query)
            return render(request, 'image_search_app/results.html', {'images': selected_images})
        return JsonResponse({"error": "No query provided."}, status=400)

    def image_search(self, query, top_k=2):
        query_tokens = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        query_embedding = vision_encoder(**query_tokens).pooler_output[0].detach().cpu().numpy()  

        similarities = np.dot(image_embeddings.cpu().numpy(), query_embedding)  
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        sorted_images = [image_paths[i] for i in top_indices]

        return sorted_images