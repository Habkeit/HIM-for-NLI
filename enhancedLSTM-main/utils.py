import torch
import torch.nn as nn
import time
from tqdm import tqdm

def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    #idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, revese_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, revese_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index

def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask

def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)

def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def unpack_nli_batch(batch, device):
    premises = batch["premises"].to(device)
    premises_lengths = batch["premise_lengths"].to(device)
    hypotheses = batch["hypotheses"].to(device)
    hypotheses_lengths = batch["hypothesis_lengths"].to(device)
    labels = batch["labels"].to(device)
    premise_word_idx = batch["premise_word_idx"].to(device)
    premise_left = batch["premise_left"].to(device)
    premise_right = batch["premise_right"].to(device)
    premise_is_leaf = batch["premise_is_leaf"].to(device)
    premise_tree_lengths = batch["premise_tree_lengths"].to(device)
    hypothesis_word_idx = batch["hypothesis_word_idx"].to(device)
    hypothesis_left = batch["hypothesis_left"].to(device)
    hypothesis_right = batch["hypothesis_right"].to(device)
    hypothesis_is_leaf = batch["hypothesis_is_leaf"].to(device)
    hypothesis_tree_lengths = batch["hypothesis_tree_lengths"].to(device)
    return (premises, premises_lengths, hypotheses, hypotheses_lengths, labels,
            premise_word_idx, premise_left, premise_right, premise_is_leaf, premise_tree_lengths,
            hypothesis_word_idx, hypothesis_left, hypothesis_right, hypothesis_is_leaf, hypothesis_tree_lengths)


def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            q1, q1_lengths, q2, q2_lengths, labels, q1_word_idx, q1_left, q1_right, q1_is_leaf, q1_tree_len, q2_word_idx, q2_left, q2_right, q2_is_leaf, q2_tree_len = unpack_nli_batch(batch, device)
            logits, probs = model(q1, q1_lengths, q2, q2_lengths, q1_word_idx, q1_left, q1_right, q1_is_leaf, q1_tree_len, q2_word_idx, q2_left, q2_right, q2_is_leaf, q2_tree_len)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy

def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            q1, q1_lengths, q2, q2_lengths, labels, q1_word_idx, q1_left, q1_right, q1_is_leaf, q1_tree_len, q2_word_idx, q2_left, q2_right, q2_is_leaf, q2_tree_len = unpack_nli_batch(batch, device)
            _, probs = model(q1, q1_lengths, q2, q2_lengths, q1_word_idx, q1_left, q1_right, q1_is_leaf, q1_tree_len, q2_word_idx, q2_left, q2_right, q2_is_leaf, q2_tree_len)
            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy

def train(model, dataloader, optimizer, criterion, epoch_number, max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    # print('train')
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    # print(tqdm_batch_iterator)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        # print(q)
        # print(h)
        # print(batch_index)
        # print((q, q_len, h, h_len, label))
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        q1, q1_lengths, q2, q2_lengths, labels, q1_word_idx, q1_left, q1_right, q1_is_leaf, q1_tree_len, q2_word_idx, q2_left, q2_right, q2_is_leaf, q2_tree_len = unpack_nli_batch(batch, device)
        optimizer.zero_grad()
        logits, probs = model(q1, q1_lengths, q2, q2_lengths, q1_word_idx, q1_left, q1_right, q1_is_leaf, q1_tree_len, q2_word_idx, q2_left, q2_right, q2_is_leaf, q2_tree_len)
        # print(logits)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy
