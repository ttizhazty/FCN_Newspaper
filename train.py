import os
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser(description = "training fcn for newpaper segmentation",
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', default = 'data/FCN_dataset',
                    help = "Directory containing dataset")
parser.add_argument('--model_dir', default = 'experiment/base_model',
                    help = "Directory contatining model")
parser.add_argument('--restore_file', default = None,
                    help = "Containing weights to reload before training")

def train(model, optimizer, loss_fn, dataloader, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        params: (Params) hyper-parameters
    Note: will add config in the future
    """
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summary = []
    loss_avg = utils.RunningAverage()

    with tqdm(total = len(dataloader)) as t:
        for i, (images, masks) in enumerate(dataloader):
            if params.cuda:
                images = image.cuda()
                masks = masks.cuda()
            
            # compute model output and loss
            output = model(images)
            loss = loss_fn(output, masks)

            # clear previous gradients, compute gradients of all variables
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Visualize the res
            # utils.show_res(output, masks)

            # evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                summary_batch = {} # saving metrics
                summary_batch['loss'] = loss.item() # can attach more metrics on summary
                summary.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    
    logging.info("- Train loss: " + str(loss_avg()))
    
    '''
    compute mean of all metrics in summary 
    TODO: add more metircs
    
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    '''
    

def train_and_eval(model, train_dataloader, val_dataloader, optimizer, loss_fn, params, 
                   model_dir, restore_file = None):
        
        # reload weights from restore_file if specified
        if restore_file is not None:
            restore_path = os.path.join(
                args.model_dir, args.restore_file + '.pth.tar')
            logging.info("Restoring parameters from {}".format(restore_path))
            utils.load_checkpoint(restore_path, model, optimizer)

        best_model_loss = 0.0

        for epoch in range(params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            # compute number of batches in one epoch (one full pass over the training set)
            train(model, optimizer, loss_fn, train_dataloader, params)

            # evaluate 
            # TODO: add valdation function



if __name__ == '__main__':
    args = parser.parse_args()

    # laod parameters
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # check GPU
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(126)
    if params.cuda:
        torch.cuda.manual_seed(126)
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl, val_dl = dataloaders['train'], dataloaders['val']
    logging.info("Done")

    # Define model optimizer and loss_fn
    model = net.FCN().cuda if params.cuda else net.FCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    
# Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_eval(model, train_dl, val_dl, optimizer, loss_fn, params, args.model_dir,
                       args.restore_file)