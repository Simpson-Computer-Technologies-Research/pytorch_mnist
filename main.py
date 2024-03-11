import torch
from torch import nn, optim


##
## Basic conv neural network. We'll use this to classify the MNIST dataset.
##
class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization


##
## Use GPU
##
device = "cuda" if torch.cuda.is_available() else "cpu"


##
## Train function
##
def train():
    ##
    ## Load the MNIST dataset
    ##
    from torchvision import datasets, transforms

    mnist_train = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    mnist_test = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )

    ##
    ## Create a DataLoader for the MNIST dataset
    ##
    from torch.utils.data import DataLoader

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

    ##
    ## Train the model
    ##
    model = ConvNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 10

    print("Training model...")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Loss: {loss.item()}")

    ##
    ## Save the model
    ##
    torch.save(model.state_dict(), "model.pth")


##
## Test function
##
def test():
    ##
    ## Load the model
    ##
    model = ConvNN().to(device)
    model.load_state_dict(torch.load("model.pth"))

    ##
    ## Load the MNIST dataset
    ##
    from torchvision import datasets, transforms

    mnist_test = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )

    ##
    ## Create a DataLoader for the MNIST dataset
    ##
    from torch.utils.data import DataLoader

    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

    ##
    ## Test the model
    ##
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred, _ = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

    from matplotlib import pyplot as plt

    # Show the first 10 images with their predicted labels and true labels
    _, ax = plt.subplots(1, 10, figsize=(10, 2))

    for i in range(10):
        ax[i].set_title(f"p:{predicted[i]},r:{y[i]}")
        ax[i].axis("off")
        ax[i].imshow(x[i].view(28, 28).cpu().numpy())

    plt.show()


##
## Main
##
if __name__ == "__main__":
    # train()
    test()
