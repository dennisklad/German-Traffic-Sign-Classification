#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import MulticlassPrecisionRecallCurve

import torchvision
import torchvision.utils
import torchvision.transforms as transforms

from PIL import Image
import time

import mmpretrain as mmp
from mmpretrain.models import backbones, classifiers, heads, necks

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm

import matplotlib.pyplot as plt


class EarlyStopping:
    """
    Early stopping callback to stop training if validation loss does not improve.
    """
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
    
    def monitor(self, model, val_loss):
        if val_loss < self.best_loss: # search for the best loss
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False


def validation(model, device, data_loader, loss_function):
    """
    Perform validation on the given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    device : torch.device
        The device on which the model and data should be loaded.
    data_loader : torch.utils.data.DataLoader
        DataLoader containing the validation dataset.
    loss_function : torch.nn.Module
        The loss function to calculate the validation loss.

    Returns
    -------
    dict
        A dictionary containing the validation metrics:
        - 'loss': Average validation loss.
        - 'outputs': Flattened predictions.
        - 'confusion_matrix': Confusion matrix.
        - 'class_precision': Precision per class.
        - 'class_recall': Recall per class.
        - 'accuracy': Overall accuracy.
        - 'f1-Macro': Macro F1-score.
        - 'report': Classification report.
    """
    model.eval() # set the validation mode
    val_loss = 0 # initlize validation loss
    correct = 0
    preds = [] # store all predictions
    targets = [] # store the ground truth

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device) # mount data to device
            outputs = model(data) # feed data to the model

            val_loss += loss_function(outputs, target).item() # running loss

            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            preds.append(pred.cpu().detach().numpy())  # store the predictions of each batch
            targets.append(target.cpu().detach().numpy()) # get the ground truth
        
            correct += pred.eq(target.view_as(pred)).sum().item() # count correct predictions
    
    epoch_loss = val_loss / len(data_loader) # avg loss

    accuracy = correct / len(data_loader.dataset) # accuracy
    
    # Flatten predictions and ground truth for further evaluation
    flattened_preds = [item for sublist in preds for item in sublist]
    flattened_targets = [item for sublist in targets for item in sublist]
    
    # Calculate confusion matrix, precision, recall, f1-score, and classification report
    cm = confusion_matrix(flattened_targets, flattened_preds) # confusion matrix
    class_precision = cm.diagonal() / cm.sum(axis=0) # precision per class
    class_recall = cm.diagonal() / cm.sum(axis=1) # recall per class
    
    f1 = f1_score(flattened_targets, flattened_preds, average='macro')
    report = classification_report(flattened_targets, flattened_preds, digits=4)

    print(f'\nValidation: Average loss: {epoch_loss:.4f}, '
          f'Accuracy: {correct}/{len(data_loader.dataset)} ({100. * accuracy:.4f}%)\n')

    
    return {'loss': epoch_loss,
            'outputs': flattened_preds,
            'confusion_matrix': cm,
            'class_precision': class_precision,
            'class_recall': class_recall,
            'accuracy': accuracy * 100,
            'f1-Macro': f1 * 100,
            'report': report,
           }
    
    
def train_val_loop(model, device, train_loader, validation_loader, 
                   optimizer, criterion, num_epochs, early_stopping=None, scheduler=None):
    
    """
    Training and validation loop for the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    device : torch.device
        The device on which the model and data should be loaded.
    train_loader : torch.utils.data.DataLoader
        DataLoader containing the training dataset.
    validation_loader : torch.utils.data.DataLoader
        DataLoader containing the validation dataset.
    optimizer : torch.optim.Optimizer
        The optimizer to update model parameters during training.
    criterion : torch.nn.Module
        The loss function to calculate the training loss.
    num_epochs : int
        Number of training epochs.
    early_stopping : EarlyStopping
        Early stopping object for monitoring validation loss.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler.

    Returns
    -------
    tuple
        A tuple containing the training and validation history.
    """
    
    model.to(device) # mount model to gpu/cpu
    
    train_history = [] # store training history
    validation_history = [] # store validation history
    training_time = 0 # initlize time
    start_time = time.time() # Start the timer
    
    for epoch in range(num_epochs):
        model.train() # set training mode       
        running_loss = 0.0 # initlize trainingloss
        preds = [] # store all predictions
        targets = [] # store the ground truth
 
        # Create a tqdm progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False, ncols=80, miniters=10)
        
        # iterate over Training data
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device) # mount images
            labels = labels.to(device) # mount labels
            optimizer.zero_grad() # reset gradient 
            outputs = model(inputs) # feed images to the model
            
            # optimization
            loss = criterion(outputs, labels) # calculate loss
            loss.backward() # backpropagation
            optimizer.step() # take gradient step
            
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            preds.append(pred.cpu().detach().numpy()) # store the predictions of each batch
            targets.append(labels.cpu().detach().numpy()) # get the ground truth
            
            running_loss += loss.item()
            
            pbar.set_postfix(loss=loss.item()) # Update the description of the progress bar with the current loss
            
        batch_loss = running_loss / len(train_loader) # avg loss per sample

        preds_flattened = [item for sublist in preds for item in sublist]
        flattened_targets = [item for sublist in targets for item in sublist]
        f1 = f1_score(flattened_targets, preds_flattened, average='macro') # F1-Score - Macro
        
        print(f"Epoch {epoch+1} - Training: Average loss: {batch_loss:.4f}")

        # store values in dict
        tr_history = {'loss': batch_loss,
                      'accuracy': f1 * 100,
                      'outputs': preds_flattened,
                     }
        train_history.append(tr_history) # Store history of all epochs
        
        # Run Validation
        with torch.no_grad():
            val_history = validation(model, device, validation_loader, criterion)
            
        validation_history.append(val_history) # Store history of all epochs
        
        if scheduler:
            scheduler.step() # Update learning rate scheduler
        
        # train-val time per epoch
        epoch_time = time.time() - start_time
        training_time += epoch_time
        print(f"Epoch {epoch+1} - Training Time: {epoch_time:.2f} seconds")
        
        # Early stopping monitoring validation loss
        if early_stopping.monitor(model, val_history['loss']):
            print("Validation loss has not improved for consecutive epochs. Early stopping.")
            break

    print(f"Epoch {epoch+1} - Validation Time: {training_time:.2f} seconds")

    return train_history, validation_history
    
    
def prec_recall(model, device, test_loader, label_names, thresholds=None):
    """
    Compute and visualize the precision-recall curves for each class.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to evaluate.
    device : torch.device
        The device on which the model and data should be loaded.
    test_loader : torch.utils.data.DataLoader
        DataLoader containing the test dataset.
    label_names : list
        List containing the names of the classes.
    thresholds : list or None, optional
        List of thresholds for binarization (default is None).

    """
    
    with torch.no_grad():
        # set up mprc object with number of classes and thresholds
        mcprc = MulticlassPrecisionRecallCurve(len(label_names), 
                                               thresholds).to(device)
        for i, (images, labels) in enumerate(test_loader):
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images) # run images through model
            mcprc(outputs, labels)  # update for batch predictions
            
    precision, recall, thresholds = mcprc.compute() # compute for each threshold in thresholds
    
    def visualize_prec_rec_curve(precision, recall, label_names):
        """
        Visualize the precision-recall curves for each class.

        Parameters
        ----------
        precision : torch.Tensor
            Tensor containing the precision values for each class at different thresholds.
        recall : torch.Tensor
            Tensor containing the recall values for each class at different thresholds.
        label_names : list
            List containing the names of the classes.

        """
        x = 10
        num_classes = len(label_names)
        fig, axes = plt.subplots(x, num_classes // x, figsize=(50, 50))
        fig.tight_layout(pad=7.0)

        for i in range(len(label_names)): # goes through columns    
            prec = precision[i, :]
            rec = recall[i, :]

            row = i // (num_classes // x) # identify row for subplot
            col = i % (num_classes // x) # identify column for subplot

            axes[row, col].plot(rec.cpu(), prec.cpu())
            axes[row, col].set_xlabel('Recall', fontsize=20)
            axes[row, col].set_ylabel('Precision', fontsize=20)
            axes[row, col].set_title(f'Class {i}: {label_names[i]}', fontsize=25)
            axes[row, col].set_xlim([0.0, 1.0])
            axes[row, col].set_ylim([0.0, 1.0])
            axes[row, col].tick_params(axis='both', labelsize=15)
        plt.show()
    
    visualize_prec_rec_curve(precision, recall, label_names) # visualize precision recall curves for each class
    
    
def get_accuracy_per_class(confusion_matrix):
    """
    Calculates the accuracy for each class based on the confusion matrix.

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        The confusion matrix containing the true positive counts for each class.

    Returns
    -------
    numpy.ndarray
        An array containing the accuracy for each class.

    """
    num_classes = confusion_matrix.shape[0]
    accuracy_per_class = np.zeros(num_classes) # initlize the acc array

    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        total_samples_class = np.sum(confusion_matrix[i])
        accuracy_per_class[i] = true_positives / total_samples_class

    return accuracy_per_class
    
    
def show_random_examples(data_loader, history, labels_dict, num_of_ex=5):
    """
    Displays random examples from the data loader along with their predicted and true labels.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The data loader containing the examples to be displayed.
    history : dict or list
        The history of model predictions (can be a dictionary or a list of dictionaries).
    labels_dict : dict
        A dictionary mapping class indices to their corresponding labels.
    num_of_ex : int, optional
        The number of random examples to be displayed. Default is 5.

    Returns
    -------
    None.

    """
    indices = random.sample(range(len(data_loader.dataset)), num_of_ex)  # Get random indices
    examples = []  # store the examples

    # Iterate over the random indices
    for i in indices:
        example, true_label = data_loader.dataset[i]  # Get the example and its true label
        if isinstance(history, list):
            predicted_label = labels_dict[history[len(history) - 1]['outputs'][i]] # get pridictions for the last epoch
        else:
            predicted_label = labels_dict[history['outputs'][i][0]] # get pridictions in case of no epochs

        examples.append((example, predicted_label, true_label)) # Add the example to the list

    # Display the examples with their labels
    fig, axs = plt.subplots(1, num_of_ex, figsize=(num_of_ex * 4, 4)) # create subplots
    for i, (example, predicted_label, true_label) in enumerate(examples):
        example_np = np.asarray(transforms.ToPILImage()(example))  # Convert tensor image to numpy array
        ex = cv2.cvtColor(example_np, cv2.COLOR_BGR2RGB) # change the color channels from BGR to RGB
        axs[i].imshow(ex) # show image in the subplot
        axs[i].set_title(f"Predicted: {predicted_label}\nTrue: {labels_dict[true_label]}") # Set title
        axs[i].axis('off')
        
        # Add green tick if the prediction matches the ground truth
        if predicted_label == labels_dict[true_label]:
            axs[i].text(-7, 5, '\u2713', fontsize=40, color='green')
        else:
            axs[i].text(-7, 5, 'X', fontsize=40, color='red')

    plt.show()    

    
def visualize_confusion_matrix(cf_matrix, labels):
    """
    Visualizes the confusion matrix as a heatmap.

    Parameters
    ----------
    cf_matrix : numpy.ndarray
        The confusion matrix to be visualized.
    labels : dict
        A dictionary mapping class indices to their corresponding labels.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(50, 40)) # Create a figure and axis
    
    # Plot confusion matrix heatmap
    cm_heatmap = sns.heatmap(10 * (cf_matrix / np.sum(cf_matrix)), ax=ax, cmap='Blues') # annot=True, fmt='.1%'

    ax.set_xlabel('Predicted', fontsize=40) # Set x-labels
    ax.set_ylabel('True', fontsize=40) # Set y-labels

    ax.set_yticklabels(labels.values(), rotation=0, fontsize=26) # Set x-ticks values
    ax.set_xticklabels(labels.values(), rotation=90, fontsize=26) # Set y-ticks values

    ax.set_title('Confusion Matrix', fontsize=45) # Set title
    
    # Adjust font size for colorbar tick labels
    cbar = cm_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    
    plt.show() # show plot    
    
    
def visualize_loss_accuracy(train_history, test_history):
    """
    This method visualizes the loss and accuracy over epochs.

    Parameters
    ----------
    train_history : list
        List of training history dictionaries.
    test_history : list
        List of testing history dictionaries.

    Returns
    -------
    None.

    """
    def plot_metric(x, train_metric, val_metric, metric_name):
        plt.figure(figsize=(8, 4))
        # Plot metrics
        if metric_name == 'Accuracy':
            sns.lineplot(x=x, y=train_metric, label='Train') # train acc
            sns.lineplot(x=x, y=val_metric, label='Validation') # val acc
        elif metric_name == 'Loss':
            sns.lineplot(x=x, y=train_metric, label='Train') # train loss
            sns.lineplot(x=x, y=val_metric, label='Validation') # val loss
        else:
            sns.lineplot(x=x, y=train_metric, label='Accuracy') # val acc
            sns.lineplot(x=x, y=val_metric, label='Loss') # val loss

        plt.xlabel('#epochs') # set x-label
        plt.ylabel(metric_name) # set y-label
        plt.legend() # show legend
        plt.xticks(x)  # show all integers
        plt.show() # show plot
        
    def normalize(myList):
        arr = np.array(myList) # Convert the list to a NumPy array
        return (arr - arr.min()) / (arr.max() - arr.min()) # Normalize to the range [0, 1]
    
    q = len(train_history) # num of epochs

    train_losses = [history['loss'] for history in train_history] # get training epochs loss
    val_losses = [history['loss'] for history in test_history] # get validation epochs loss
    train_accuracy = [history['accuracy'] for history in train_history] # get training epochs acc 
    val_accuracy = [history['accuracy'] for history in test_history] # get validation epochs acc 

    plot_metric(range(1, q + 1), normalize(val_accuracy), normalize(val_losses), 'Validation') # Val-Acc vs. Val-Acc
    plot_metric(range(1, q + 1), train_accuracy, val_accuracy, 'Accuracy') # Train- vs. Val-accuracy
    plot_metric(range(1, q + 1), train_losses, val_losses, 'Loss') # Train- vs. Val-loss

