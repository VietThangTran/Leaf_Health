import torch, PIL
import os

class Leaf_Dataset(torch.utils.data.Dataset):
  def __init__(self,root_path,df, transform = None):
    self.root_path = root_path
    self.df = df
    self.images = self._get_images()
    self.transform = transform

  def _get_images(self):
    images = []
    for row in self.df.iloc:
      name = row['image:FILE']
      label = row['category']
      image_path = os.path.join(self.root_path,name)
      images.append((image_path, label))
    return images

  def __len__(self):
    return len(self.df)

  def __getitem__(self,idx):
    img_path, label = self.images[idx]
    img = PIL.Image.open(img_path)
    if self.transform:
      img = self.transform(img)
    return img, label