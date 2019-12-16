import torch
from scipy.sparse import lil_matrix

def signal_entries(input, output):
    '''
    This function takes 2 input matrices of the same size (x and y layer for a certain detector) and returns the subset of the output matrix corresponding to the possible signal points (match between x and y layers)
    :param tuple input: (tensor ymatrix, tensor xmatrix)
    :param torch tensor output: layer matrix
    :return: (torch tensor) subset od output corresponding to possible signal points
    '''
    ymatrix = input[0]
    xmatrix = input[1]
    ymatrix = ymatrix[:, 0] # take only one column
    xmatrix = xmatrix[0, :] # take only one row
    y_index = ymatrix.nonzero(as_tuple=True)[0]
    x_index = xmatrix.nonzero(as_tuple=True)[0]
    target = torch.index_select(output, 0, y_index)
    target = torch.index_select(target, 1, x_index)
    return target

def accuracy_step(prediction, target):
    corr_tensor = torch.eq(prediction, target)
    #print(corr_tensor)
    correct = float(corr_tensor.sum())
    total = corr_tensor.numel()
    #print(total)
    return correct, total

def f1_step(prediction, target):
    true_tensor = torch.ones(tuple(prediction.size()))
    false_tensor = torch.zeros(tuple(prediction.size()))
    true_positives = float((prediction * target).sum())
    # print(true_positives)
    reversed_pred = torch.where(prediction == 0, true_tensor, false_tensor)
    reversed_targ = torch.where(target == 0, true_tensor, false_tensor)
    false_negatives = float((reversed_pred * target).sum())
    false_positives = float((prediction * reversed_targ).sum())
    actual_positives = true_positives + false_negatives
    pred_positives = true_positives + false_positives
    true_negatives = float((reversed_pred*reversed_targ).sum())
    return true_positives, actual_positives, pred_positives, true_negatives

def transf_prediction(prediction, thr):
    true_tensor = torch.ones(tuple(prediction.size()))
    false_tensor = torch.zeros(tuple(prediction.size()))
    tr_pred = torch.where(prediction >= thr, true_tensor, false_tensor)
    return tr_pred

def val_loop(model, data_loader, path_rundir, use_cuda=True, calc_metrics=True):
    vloss_filename = '{}/val_losses.csv'.format(path_rundir)
    f_loss = open(vloss_filename, 'w+')
    corr_overall = 0 # Correct predictions
    tot_overall = 0 # Total number of predictions
    tp_overall = 0 # True positives
    ap_overall = 0 # Actual number of positives
    pp_overall = 0 # Predicted number of positives
    tn_overall = 0 # True negatives
    with torch.no_grad():
        for j, val_data in enumerate(data_loader, 0):
            val_local_datapoint, val_local_target = val_data
            if use_cuda and torch.cuda.is_available():
                val_local_datapoint = val_local_datapoint.cuda()
                val_local_target = val_local_target.cuda()
            model.eval()

            val_prediction = model(val_local_datapoint.float())
            # Calculate the loss and print it to file
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum', weight=mask(val_local_datapoint.float()))
            val_loss = loss_fn(val_prediction.float(), val_local_target.float())
            f_loss.write('{},'.format(val_loss.item()))
            # print(val_loss.item())

            if calc_metrics:
                # Calculate accuracy, precision, recall and f1 of the model
                for layer in range(6):
                    for sample in range(tuple(val_prediction.size())[0]):
                        input = (val_local_datapoint[sample, layer, :, :], val_local_datapoint[sample, 6+layer, :, :])
                        pred = signal_entries(input, val_prediction[sample, layer, :, :])
                        # print(pred)
                        pred = transf_prediction(pred, 0)
                        # print(torch.nonzero(pred, as_tuple=True))
                        targ = signal_entries(input, val_local_target[sample, layer, :, :])
                        #print(targ.size())
                        if pred.nelement() > 1: # Checking only in case of ambiguity
                            corr, tot = accuracy_step(pred, targ)
                            corr_overall += corr
                            tot_overall += tot
                            true_pos, actual_pos, pred_pos, true_neg = f1_step(pred, targ)
                            tp_overall += true_pos
                            ap_overall += actual_pos
                            pp_overall += pred_pos
                            tn_overall += true_neg

    f_loss.close()

    if calc_metrics:
        accuracy = corr_overall/tot_overall # Accuracy
        print('Accuracy:', accuracy)
        rec = tp_overall/ap_overall # Recall
        print('Recall:', rec)
        if pp_overall>0:
            prec = tp_overall/pp_overall # Precision
            f1 = 2 * prec * rec / (prec + rec)
            print('Precision:', prec)
            print('f1:', f1)
        else:
            print('Zero predicted positives')
        print('True negatives:', tn_overall)

def mask(input):
    grid_dim = 512
    batches = tuple(input.size())[0]
    mask = torch.zeros([batches, 6, grid_dim, grid_dim])
    for layer in range(6):
        for sample in range(batches): # loop over batch dimension
            ymatrix = input[sample, layer, :, :]
            xmatrix = input[sample, 6 + layer, :, :]
            true_tensor = torch.ones(tuple(ymatrix.size()))
            false_tensor = torch.zeros(tuple(ymatrix.size()))
            prod = ymatrix*xmatrix
            m = torch.where(prod!=0, true_tensor, false_tensor)
            mask[sample, layer, :, :] = m
    return mask
