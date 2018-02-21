import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from ranker import DualLstmRanker, TrainData, DevData
import utils, constants

def main():
    # Load word index
    word_to_idx = utils.read_data_pkl(constants.WORD_INDEX_FILE)

    # Create data and model objects
    train_data = TrainData()
    train_dloader = DataLoader(train_data, batch_size=constants.BATCH_SIZE, shuffle=True, num_workers=1)
    dev_data = DevData()
    dev_dloader = DataLoader(dev_data, batch_size=constants.DEV_BATCH_SIZE, shuffle=False, num_workers=1)
    model = DualLstmRanker(len(word_to_idx))

    if constants.ONGPU:
        model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)

    for iter in range(constants.NUM_EPOCHS):
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for i_batch, (c, r, y) in enumerate(train_dloader):
            if constants.ONGPU:
                c, r, y = Variable(c.cuda(), requires_grad=False), \
                          Variable(r.cuda(), requires_grad=False), \
                          Variable(y.cuda(), requires_grad=False)
            else:
                c, r, y = Variable(c, requires_grad=False), \
                          Variable(r, requires_grad=False), \
                          Variable(y, requires_grad=False)
            y = y.unsqueeze(1).float()
            print(c.size(), r.size())
            optimizer.zero_grad()
            logits = model.forward(c, r)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            print(logits, F.sigmoid(logits))
            running_loss += loss.data[0]
            acc = utils.get_accuracy(F.sigmoid(logits), y)
            running_acc += acc
            print(loss.data[0], acc)
        print("Train: Epoch: %d Loss: %.3f " % (
            iter, running_loss / len(train_dloader)))
        print("----------------------------------------------")

        model.eval()
        recall = 0
        for d_batch, (c_dev, r_dev, y_dev) in enumerate(dev_dloader):
            if constants.ONGPU:
                c_dev, r_dev, y_dev = Variable(c_dev.cuda(), requires_grad=False), \
                                      Variable(r_dev.cuda(), requires_grad=False), \
                                      Variable(y_dev.cuda(), requires_grad=False)
            else:
                c_dev, r_dev, y_dev = Variable(c_dev, requires_grad=False), \
                                      Variable(r_dev, requires_grad=False), \
                                      Variable(y_dev, requires_grad=False)

            #TODO - Optimize by encoding context just once
            logits = model.forward(c_dev, r_dev)
            probs = F.sigmoid(logits)
            probs = probs.split(constants.NUM_DISTRACTORS)

            y_true = [0 for _ in range(len(probs))]
            y_preds = []
            for p in probs:
                _, preds = p.sort(descending=True)
                y_preds.append((preds))
            recall += utils.evaluate_recall(y_preds, y_true)
        print(recall/len(dev_dloader))


if __name__ == "__main__":
    main()