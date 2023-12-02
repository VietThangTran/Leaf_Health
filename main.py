import Leaf_Dataset, Leaf_DataFrame, Leaf_Dataloader, Leaf_Health_Model
import torchvision, torch

if __name__ == '__main__':
    data_dir = 'data/val.csv'

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((200,200)),
                                                         torchvision.transforms.ToTensor()])
    
    df = Leaf_DataFrame.Leaf_DataFrame(data_dir)
    dataset = Leaf_Dataset.Leaf_Dataset('data',df.get_df(), transform)
    dataloader = Leaf_Dataloader.Leaf_Dataloader(dataset,batch_size=16, shuffle= True).get_dataloader()
    pretrained_model = torch.load('Leaf_Health.pth')
    model = Leaf_Health_Model.Leaf_Health_Model()
    model.load_state_dict(pretrained_model['model'])

    confusion_matrix = model.confusion_matrix(val_dataloader=dataloader)
    print(confusion_matrix)