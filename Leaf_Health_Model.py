import torch, torchvision

class Leaf_Health_Model(torchvision.models.DenseNet):
  def __init__(self):
    super(Leaf_Health_Model,self).__init__()
    pretrained_weigth = torchvision.models.densenet121(pretrained = True).state_dict()
    self.load_state_dict(pretrained_weigth)
    self.classifier = torch.nn.Linear(1024,3, bias = True)
    self.criterion = torch.nn.CrossEntropyLoss()
    self.optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)

  def forward(self,X):
    o = super(Leaf_Health_Model,self).forward(X)
    return o
  
  def trainer(self,epochs, dataloader):
    for epoch in range(epochs):
        for img, label in dataloader:
            o = self.forward(img)

            target = torch.tensor(label)
            loss = self.criterion(o, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    return None
  
  def evaluate(self,val_dataloader):
    result = []
    labels = []
    for img, label in val_dataloader:
        o = self.forward(img)

        result += o.argmax(dim = -1).tolist()
        labels += label.tolist()

    return result, labels
  
  def confusion_matrix(self,val_dataloader):
    result, labels = self.evaluate(val_dataloader)
    confusion_matrix = [[0 for _ in range(3)] for _ in range(3)]
    for pred, target in zip(result,labels):
        confusion_matrix[pred][target] += 1
    return confusion_matrix

