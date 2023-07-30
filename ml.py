import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid

from models import SimpleModel

NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

def train(model, dataloader, val_dataloader, loss_function, optimizer):
    """
    trains the model
    """
    def training_step():
        model.train()
        training_loss, training_acc = 0, 0
        for batch_num, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            #forward pass
            predictions = model(images)
            loss = loss_function(predictions, labels)
            training_loss += loss.item()
            #backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_class = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
            training_acc += (pred_class == labels).sum().item() / len(predictions)
        training_loss /= len(dataloader)
        training_acc /= len(dataloader)
        return training_loss, training_acc 

    results = {
        "training_loss" : [],
        "training_acc" : [],
        "validation_loss" : [],
        "validation_acc": []
    }
    for epoch in range(NUM_EPOCHS):
        l, a = training_step()
        results["training_loss"].append(l)
        results["training_acc"].append(a)
        vl, va = validate(model, val_dataloader, loss_function)
        results["validation_loss"].append(vl)
        results["validation_acc"].append(va)
        print(
            f"epoch {epoch}\n"
            f"\tTraining loss: {l}\n"
            f"\tTraining acc: {a}\n"
            f"\tValidation loss: {vl}\n"
            f"\tvalidation acc: {va}")

    return results


def validate(model , dataloader, loss_function):
    """
    validates the model
    """
    model.eval()
    total_loss, total_acc = 0, 0
    correct=0
    with torch.inference_mode():
        for batch_num, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            #forward pass
            predictions = model(images)
            loss = loss_function(predictions, labels)
            total_loss += loss.item()
            predicted_classes = torch.argmax(torch.softmax(predictions, dim=1), dim=1) # this is from a softmax which converts the logit to a probability for each class, then the argmax picks the index of highest probability and returns that
            total_acc += (labels == predicted_classes).sum().item() / len(predictions)
        total_loss /= len(dataloader)
        total_acc /= len(dataloader)
    return total_loss, total_acc


def plot_loss(results : dict):
    epochs = [i for i in range(NUM_EPOCHS)]
    plt.plot(epochs, results['training_loss'], label = "training loss")
    plt.plot(epochs, results['validation_loss'], label='validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def test(model, dataloader, classes):
    """
    tests the model on a small test set and displays results
    """
    fig = plt.figure(figsize=(25, 8))
    model.eval()
    total_acc = 0
    with torch.inference_mode():
        for batch_num, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            preds = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
            ax = fig.add_subplot(4, 5, batch_num + 1, xticks=[], yticks=[])
            image = images[0].permute(1, 2, 0).cpu()
            plt.imshow(image)
            ax.set_title(f"Pred: {classes[preds[0]]}", color = "green" if preds[0] == labels[0] else "red")
            total_acc += (labels == preds).sum().item() / len(predictions) 
    print(f"Test Accuracy: {total_acc / len(dataloader)}")   
    plt.show()


def transfer_vgg(device):
    model = models.vgg16(pretrained=True)
    # DO NOT train the convolutional part of VGG
    for param in model.features.parameters():
        param.requires_grad = False
    # we have two classes, so the output should have two nodes
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    model.to(device)
    return model


if __name__ == "__main__":

    # use gpu if available
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

    # get directories of separated data
    train_dir = Path("./data/train")
    validation_dir = Path("./data/validate")
    test_dir = Path("./data/test")

    # create transforms to prepare images
    train_transform = transforms.Compose([
        transforms.Resize(size = (224, 224)),
        transforms.RandomHorizontalFlip(p = 0.5),  
        #TODO: subtract mean RGB value of dataset from each pixel
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.Resize(size = (224, 224)),
        #TODO: subtract mean RGB value of dataset from each pixel
        transforms.ToTensor()])

    # create datasets
    train_data = datasets.ImageFolder(
        root = train_dir, 
        transform = train_transform, 
        target_transform = None)

    validation_data = datasets.ImageFolder(
        root = validation_dir,
        transform = test_transform,
        target_transform = None)

    test_data = datasets.ImageFolder(
        root = test_dir,
        transform = test_transform,
        target_transform = None)

    # create dataloaders
    train_dataloader = DataLoader(
        dataset = train_data,
        batch_size = 16,
        num_workers = 4, 
        shuffle = True)

    validation_dataloader = DataLoader(
        dataset = validation_data,
        batch_size = 1,
        num_workers = 1, 
        shuffle = False)

    test_dataloader = DataLoader(
        dataset = test_data,
        batch_size = 1,
        num_workers = 1, 
        shuffle = False)

    # instantiate model
    #model = SimpleModel().to(device)
    model = transfer_vgg(device)

    # specify loss function
    loss_function = nn.CrossEntropyLoss()

    #specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # train model
    results = train(model, train_dataloader, validation_dataloader, loss_function, optimizer)

    # plot loss 
    plot_loss(results)

    # test model on test set
    classes = test_data.classes
    test(model, test_dataloader, classes)
