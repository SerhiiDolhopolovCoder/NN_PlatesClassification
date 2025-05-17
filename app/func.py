from timeit import default_timer as timer

import torch
from tqdm import tqdm

from config import device


def train(train_dataloader, 
          valid_dataloader,
          model, 
          loss_fn, 
          accuracy_fn, 
          optimizer,  
          epochs: int, 
          scheduler = None):
    start_timer = timer()
    total_train_loss = []
    total_train_accuracy = []
    total_valid_loss = []
    total_valid_accuracy = []
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(train_dataloader,
                                                model, 
                                                loss_fn, 
                                                accuracy_fn, 
                                                optimizer)
        total_train_loss.append(train_loss)
        total_train_accuracy.append(train_accuracy)
        
        valid_loss, valid_accuracy = valid_step(valid_dataloader,
                                                model, 
                                                loss_fn, 
                                                accuracy_fn)
        total_valid_loss.append(valid_loss)
        total_valid_accuracy.append(valid_accuracy)
        print(f"Epoch: {epoch+1}/{epochs} | ",
              f"Train loss: {train_loss:.4f} | ",
              f"Train accuracy: {train_accuracy:.2f} | ",
              f"Valid loss: {valid_loss:.4f} | ",
              f"Valid accuracy: {valid_accuracy:.2f}")
        if scheduler:
            scheduler.step(valid_loss)

    end_timer = timer()
    print(f"Training time: {end_timer - start_timer:.2f} seconds")
    
def train_step(train_dataloader, 
               model, 
               loss_fn, 
               accuracy_fn, 
               optimizer, 
               ) -> tuple[float, float]:
    """
    Returns:
        tuple[float, float]: total loss, total accuracy
    """
    train_total_loss = 0
    train_total_accuracy = 0
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_total_loss += loss.item()
        accuracy = accuracy_fn.to(device)(y, y_pred.argmax(dim=1))
        train_total_accuracy += accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f"Batch: {batch+1}/{len(train_dataloader)} | ",
        #       f"Train loss: {loss.item():.4f} | ",
        #       f"Train accuracy: {accuracy:.2f}")
        
    train_total_loss /= len(train_dataloader)
    train_total_accuracy /= len(train_dataloader)
    return train_total_loss, train_total_accuracy

def valid_step(valid_dataloader, 
              model, 
              loss_fn, 
              accuracy_fn) -> tuple[float, float]:
    """
    Returns:
        tuple[float, float]: total loss, total accuracy
    """
    valid_total_loss = 0
    valid_total_accuracy = 0
    model.eval()
    with torch.inference_mode():
        for X, y in valid_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            valid_total_loss += loss.item()
            accuracy = accuracy_fn.to(device)(y, y_pred.argmax(dim=1))
            valid_total_accuracy += accuracy
        valid_total_loss /= len(valid_dataloader)
        valid_total_accuracy /= len(valid_dataloader)
    return valid_total_loss, valid_total_accuracy

