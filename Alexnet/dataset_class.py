import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset

#Create pytorch dataset from given root directory; Similar to flow_from_directory in keras
class Dataset(Dataset):
   def __init__(self, root):
       self.path_list = []
       self.label_list = []
       count = 0
       dir_list = [x[1] for x in os.walk(root)][0]
       labels = np.arange(len(dir_list))
       label_dict = {dir_list[i]:labels[i] for i in range(len(dir_list))}
       for a in dir_list:
           d = os.path.join(root,a)
           for path in os.listdir(d):
                if os.path.isfile(os.path.join(d, path)):
                    count += 1
                    self.path_list.append(os.path.join(d, path))
                    self.label_list.append(label_dict[a])
       self.length = count

   def __len__(self):
       return self.length

   def __getitem__(self, index):
       img = Image.open(self.path_list[index])
       #Image size of 224 X 224 as input
       transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
       x = transform(img)
       y = self.label_list[index]

       return x, y
