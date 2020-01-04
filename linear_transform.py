from Datasets import ZSLDataset
from torch.utils.data import DataLoader
import torch
from torch import optim

AwA_train_1 = ZSLDataset(dataset_path='Data/AwA2/train_set_1')
AwA_test = ZSLDataset(dataset_path='Data/AwA2/test_set')
# model = torch.nn.Sequential(torch.nn.Linear(2048, 1000),
#                             torch.nn.Linear(1000, 500),
#                             torch.nn.Linear(500, 300))
model = torch.nn.Sequential(torch.nn.Linear(2048, 300))
criterion = torch.nn.MSELoss()  # L2 - Loss
optimizer = optim.Adam(model.parameters(), 0.01)

trainloader = DataLoader(AwA_train_1, batch_size=128, shuffle=True)
testloader = DataLoader(AwA_test, batch_size=64, shuffle=True)

epochs = 10
for e in range(epochs):
    running_loss = 0
    for sample in trainloader:
        # Get data
        im_embedding = sample['image_embedding'].float()
        word_embedding = sample['class_embedding'].float()

        # Training pass
        optimizer.zero_grad()

        output = model(im_embedding)
        loss = criterion(output, word_embedding)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f'Training loss: {running_loss/len(trainloader)}')
        test_loss = 0
        with torch.no_grad():
            for test_sample in testloader:
                # Get image embedding
                im_embedding_test = test_sample['image_embedding'].float()
                word_embedding_test = test_sample['class_embedding'].float()

                test_out = model(im_embedding_test)
                test_batch_loss = criterion(test_out, word_embedding_test)

                test_loss += test_batch_loss.item()
            else:
                print(f'Test error: {test_loss / len(testloader)}')
