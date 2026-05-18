from sys import platform
import torch
import torch.nn as nn
from utils import sort_by_seq_lens, masked_softmax, weighted_sum

class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """
    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.
        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).
        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


class Seq2SeqEncoder(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=False):
        "rnn_type must be a class inheriting from torch.nn.RNNBase"
        assert issubclass(rnn_type, nn.RNNBase)
        super(Seq2SeqEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.encoder = rnn_type(input_size, hidden_size, num_layers, bias=bias,
                                batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
        outputs, _ = self.encoder(packed_batch, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # for linux
        if platform == "linux" or platform == "linux2":
            reordered_outputs = outputs.index_select(0, restoration_idx)
        # for win10
        else:
            reordered_outputs = outputs.index_select(0, restoration_idx.long())
        return reordered_outputs

class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """
    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())
        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)
        return attended_premises, attended_hypotheses


class BinaryTreeLSTM(nn.Module):
    """
    Binary Tree-LSTM with independent forget gates for left and right children.
    """
    def __init__(self, embedding_dim, hidden_size):
        super(BinaryTreeLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # For leaves
        self.W_i_leaf = nn.Linear(embedding_dim, hidden_size)
        self.W_o_leaf = nn.Linear(embedding_dim, hidden_size)
        self.W_u_leaf = nn.Linear(embedding_dim, hidden_size)

        # For internal nodes
        self.W_i_int = nn.Linear(2 * hidden_size, hidden_size)
        self.W_fl_int = nn.Linear(2 * hidden_size, hidden_size)
        self.W_fr_int = nn.Linear(2 * hidden_size, hidden_size)
        self.W_o_int = nn.Linear(2 * hidden_size, hidden_size)
        self.W_u_int = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, word_idx_batch, left_batch, right_batch, is_leaf_batch, embeddings):
        batch_size, max_nodes = word_idx_batch.shape
        device = word_idx_batch.device

        h = torch.zeros(batch_size, max_nodes, self.hidden_size, device=device)
        c = torch.zeros(batch_size, max_nodes, self.hidden_size, device=device)

        for i in range(max_nodes):
            for b in range(batch_size):
                if is_leaf_batch[b, i]:
                    emb = embeddings(word_idx_batch[b, i])
                    i_gate = torch.sigmoid(self.W_i_leaf(emb))
                    o_gate = torch.sigmoid(self.W_o_leaf(emb))
                    u = torch.tanh(self.W_u_leaf(emb))
                    c[b, i] = i_gate * u
                    h[b, i] = o_gate * torch.tanh(c[b, i])
                else:
                    l = left_batch[b, i]
                    r = right_batch[b, i]
                    if l >= 0 and r >= 0:
                        h_l = h[b, l].clone()
                        h_r = h[b, r].clone()
                        c_l = c[b, l].clone()
                        c_r = c[b, r].clone()
                        x = torch.cat([h_l, h_r], dim=-1)
                        i_gate = torch.sigmoid(self.W_i_int(x))
                        f_l_gate = torch.sigmoid(self.W_fl_int(x))
                        f_r_gate = torch.sigmoid(self.W_fr_int(x))
                        o_gate = torch.sigmoid(self.W_o_int(x))
                        u = torch.tanh(self.W_u_int(x))
                        c[b, i] = i_gate * u + f_l_gate * c_l + f_r_gate * c_r
                        h[b, i] = o_gate * torch.tanh(c[b, i])

        return h
