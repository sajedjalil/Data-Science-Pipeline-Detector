import numpy as np
import argparse
import torch
import torch.nn as nn
from custom_dataset import get_train_val_loaders, get_test_loader
from preprocess_data import get_data_frames, get_vocab_and_embed_matrix
from model import Net
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


def train_epoch(model, data_loader, device, criterion, optimizer, scheduler, num_samples):
    model.train()

    losses = []
    correct = 0

    for batch_idx, (q1, q2, targets) in enumerate(data_loader):
        q1 = q1.to(device)
        q2 = q2.to(device)
        targets = targets.to(device)

        output = model(q1.long(), q2.long())
        output = output.squeeze()
        loss = criterion(output.float(), targets.float())

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        output = output >= 0.5
        correct += (output == targets).sum().item()

    acc = (correct * 1.0) / num_samples

    scheduler.step(acc)

    return acc, np.mean(losses)


def val_epoch(model, data_loader, device, criterion, num_samples):
    model.eval()

    losses = []
    correct = 0

    with torch.no_grad():
        for batch_idx, (q1, q2, targets) in enumerate(data_loader):
            q1 = q1.to(device)
            q2 = q2.to(device)
            targets = targets.to(device)

            output = model(q1.long(), q2.long())
            output = output.squeeze()
            loss = criterion(output.float(), targets.float())

            losses.append(loss.item())

            output = output >= 0.5
            correct += (output == targets).sum().item()

    return (correct * 1.0) / num_samples, np.mean(losses)


def train(model, EPOCHS, device, train_loader, val_loader, criterion, optimizer, scheduler, idx):
    history = defaultdict(list)
    best_val_acc = 0
    for _ in tqdm(range(EPOCHS)):
        train_acc, train_loss = train_epoch(model, train_loader, device, criterion, optimizer, scheduler,
                                            len(train_loader.sampler))
        val_acc, val_loss = val_epoch(model, val_loader, device, criterion, len(val_loader.sampler))

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{idx}.pth.tar')
    return history, best_val_acc


def test_model(model, data_loader, device):
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch_idx, (q1, q2) in enumerate(tqdm(data_loader)):
            q1 = q1.to(device)
            q2 = q2.to(device)

            output = model(q1.long(), q2.long())  # (batch_size, 1)
            output = output.squeeze()
            predictions.append(output)

    return predictions


def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', type=int, nargs='+', default=2)
#     parser.add_argument('--batch_size', type=int, nargs='+', default=32)
#     parser.add_argument('--learning_rate', type=float, nargs='+', default=1e-2)
#     parser.add_argument('--optim_name', type=str, nargs='+', default='adam')
#     parser.add_argument('--num_workers', type=int, default=0)
#     parser.add_argument('--gpuidx', type=int, default=0)
#     parser.add_argument('--train_val_split', type=float, default=0.8)
#     parser.add_argument('--embed_dim', type=int, default=200)
#     parser.add_argument('--sample_rows_per_class', type=int, default=10000)
#     parser.add_argument('--pretrained_embed', type=int, default=1)

#     args = parser.parse_args()

#     EPOCHS_LIST = args.epochs
#     BATCH_SIZE_LIST = args.batch_size
#     LEARNING_RATE_LIST = args.learning_rate
#     OPTIM_NAME_LIST = args.optim_name
#     NUM_WORKERS = args.num_workers
#     GPUIDX = args.gpuidx
#     TRAIN_VAL_SPLIT = args.train_val_split
#     EMBED_DIM = args.embed_dim
#     SAMPLE_ROWS_PER_CLASS = args.sample_rows_per_class
#     PRETRAINED_EMBED = args.pretrained_embed

    
    EPOCHS_LIST = [70]
    BATCH_SIZE_LIST = [32]
    LEARNING_RATE_LIST = [2e-3]
    OPTIM_NAME_LIST = ['sgd']
    NUM_WORKERS = 4
    GPUIDX = 0
    TRAIN_VAL_SPLIT = 0.8
    EMBED_DIM = 100
    SAMPLE_ROWS_PER_CLASS = 140000
    PRETRAINED_EMBED = 1


    DEVICE = f'cuda:{GPUIDX}' if torch.cuda.is_available() else 'cpu'

    train_df, test_df, sample_sub_df = get_data_frames(sample_rows_per_class=SAMPLE_ROWS_PER_CLASS)
    vocab, embed_matrix = get_vocab_and_embed_matrix(train_df, EMBED_DIM)
    print(f'Vocab size: {len(vocab.vocab)}')

    test_loader = get_test_loader(vocab,
                                  test_df,
                                  64,
                                  NUM_WORKERS,
                                  )

    best_overall_acc = 0
    idx = 0
    optimal_hyperparams = defaultdict(int)

    vocab_size = len(vocab.vocab)
    model = Net(vocab_size, embed_matrix, pretrained_embed=PRETRAINED_EMBED).to(DEVICE)
    criterion = nn.BCELoss()

    total_grid_search_params = len(EPOCHS_LIST) * len(BATCH_SIZE_LIST) * len(LEARNING_RATE_LIST) * len(OPTIM_NAME_LIST)

    for EPOCHS in EPOCHS_LIST:
        for BATCH_SIZE in BATCH_SIZE_LIST:
            for LEARNING_RATE in LEARNING_RATE_LIST:
                for OPTIM_NAME in OPTIM_NAME_LIST:
                    idx += 1
                    print('-' * 10)
                    print(f'Grid Search: {idx}/{total_grid_search_params}')
                    print('-' * 10)
                    train_loader, val_loader = get_train_val_loaders(vocab,
                                                                     train_df,
                                                                     BATCH_SIZE,
                                                                     TRAIN_VAL_SPLIT,
                                                                     NUM_WORKERS,
                                                                     )

                    optimizer = None
                    if OPTIM_NAME == 'sgd':
                        optimizer = torch.optim.SGD(model.parameters(),
                                                    lr=LEARNING_RATE,
                                                    momentum=0.9,
                                                    nesterov=True)
                    if OPTIM_NAME == 'adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                           mode='max',
                                                                           factor=0.1,
                                                                           patience=2,
                                                                           verbose=False)

                    history, best_val_acc = train(model, EPOCHS, DEVICE, train_loader, val_loader, criterion, optimizer,
                                                  scheduler, idx)

                    if best_val_acc > best_overall_acc:
                        best_overall_acc = best_val_acc
                        optimal_hyperparams['epochs'] = EPOCHS
                        optimal_hyperparams['batch_size'] = BATCH_SIZE
                        optimal_hyperparams['learning_rate'] = LEARNING_RATE
                        optimal_hyperparams['optim_name'] = OPTIM_NAME
                        optimal_hyperparams['idx'] = idx

                        plt.figure()
                        plt.plot(history['train_loss'], label='train loss')
                        plt.plot(history['val_loss'], label='val loss')
                        plt.title('Training Loss vs Validation Loss')
                        plt.ylabel('Loss')
                        plt.xlabel('Epoch')
                        plt.legend()
                        plt.savefig(f'best_quora_train_val_loss.png')

                        plt.figure()
                        plt.plot(history['train_acc'], label='train acc')
                        plt.plot(history['val_acc'], label='val acc')
                        plt.title('Training Acc vs Validation Acc')
                        plt.ylabel('Acc')
                        plt.xlabel('Epoch')
                        plt.legend()
                        plt.savefig(f'best_quora_train_val_acc.png')

    print('-' * 20)
    print('-' * 20)
    print(f'Optimal hyperparameters:'
          f'\nepochs={optimal_hyperparams["epochs"]}'
          f'\nbatch_size={optimal_hyperparams["batch_size"]}'
          f'\nlearning_rate={optimal_hyperparams["learning_rate"]}'
          f'\noptim_name={optimal_hyperparams["optim_name"]}')
    print(f'Best Validation Acc for optimal hyperparameters: {best_overall_acc}')
    print('-' * 20)
    print('Testing')

    # Load best model
    optimal_idx = optimal_hyperparams['idx']
    model.load_state_dict(torch.load(f'best_model_{optimal_idx}.pth.tar'))
    predictions = test_model(model, test_loader, DEVICE)
    preds = torch.cat(predictions)
    preds = preds.cpu().numpy()
    sample_sub_df['is_duplicate'] = preds
    sample_sub_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()
