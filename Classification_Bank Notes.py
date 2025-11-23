import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scale_inputs(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values) #scaler should be fitted only on training data, then applied to test data.
    return X_train_scaled, X_test_scaled

'''Define a simple neural network'''
class SimpleNN(nn.Module):
    def __init__(self, n_features):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n_features,100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50,1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_model(model, criterion, optimizer, X_train, y_train, epochs):
    model.train()
    losses = [] #put loss into array

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def test_model(model, criterion, X_test, y_test):
    model.eval() #set mode to eval mode
    with torch.no_grad():
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
    return loss.item()


def show_diffs(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        pred_probs = torch.sigmoid(logits) #converts to probablities
        #determines the final predicted class, [0, 1]
        pred_labels = (pred_probs > 0.5).float() #If the probability is > 0.5, predict 1, If the probability is ≤ 0.5, predict 0

        #in case use gpu
        output = torch.cat((X_test.cpu(), 
                            y_test.cpu(),
                            pred_probs.cpu(), 
                            pred_labels.cpu()), dim=1)

        #prob displays how sure the model is, say 0.97. Predicted tells the final decision in form of 0 and 1.
        df = pd.DataFrame(data=output.numpy(),
                        columns=['Length','Left','Right' ,'Bottom' ,
                        'Top' , 'Diagonal', "Actual", "Prob", "Predicted"])
        print(df.sort_index())

    correct = (pred_labels == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct/total
    print(f"Test Accuracy: {accuracy*100:.2f}%")

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Loss curve')
    plt.show()

'''Main prog'''

df = pd.read_csv("banknotes.csv")

#print(df)

#of 0 denotes a Counterfeit sample, and a value of 1 denotes a Genuine sample
#[length ,Left ,Right ,Bottom ,Top , Diagonal]

X_train = df[['Length','Left','Right' ,'Bottom' ,'Top' , 'Diagonal']]

y_train = df['Genuine']

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42, shuffle=True)

# Save original indices for later reference
X_test_indices = X_test.index

print(X_train.head())
print(y_train.head())
print(X_train.shape)
print(y_train.shape)

'''Scale features'''
X_train_scaled, X_test_scaled = scale_inputs(X_train, X_test)

'''Train'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Converts df values to float32 tensor
X_train = torch.FloatTensor(X_train_scaled).to(device)
y_train = torch.FloatTensor(y_train.values).reshape(-1,1).to(device)

X_test = torch.FloatTensor(X_test_scaled).to(device)
y_test = torch.FloatTensor(y_test.values).reshape(-1,1).to(device)

#create the neural network model
model = SimpleNN(n_features=X_train.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0005)

print(model)

#train model
print("Training the model...")
bce_loss_train = train_model(model, criterion, optimizer, X_train, y_train, epochs=500)
plot_loss(bce_loss_train)

#test model
print("Testing the model...")
bce_loss_test = test_model(model, criterion, X_test, y_test)
print(f'Test Loss (BCE): {bce_loss_test:.6f}\n') #lower the better

#show predictions vs actual values
show_diffs(model, X_test, y_test)

#counterfeit notes have probabilities near 0.0
#genuine notes have probabilities near 1.0

