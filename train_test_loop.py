import time
import torch
def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_model(model: torch.nn.Module, 
                train_data: torch.utils.data.DataLoader, 
                test_data: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                epochs: int) -> torch.nn.Module:
    start_time = time.time()

    for epoch in range(epochs):
        print(f'-----------Epoch {epoch+1}/{epochs}-----------')
        train_loss = 0
        for batch, (X, y) in enumerate(train_data):
            model.train()
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_data)

        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X, y in test_data:
                y_pred = model(X)
                test_loss += loss_fn(y_pred, y).item()
                test_acc += accuracy(y_true=y, y_pred=torch.argmax(y_pred, dim=1))

        test_loss /= len(test_data)
        test_acc /= len(test_data)
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss: .4f} | Test Accuracy: {test_acc:.2f}%")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")  
    return model

def test_model(model: torch.nn.Module, 
                test_data: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module) -> None:
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_data:
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            test_acc += accuracy(y_true=y, y_pred=torch.argmax(y_pred, dim=1))
        test_loss /= len(test_data)
        test_acc /= len(test_data)

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc: .2f}%")
