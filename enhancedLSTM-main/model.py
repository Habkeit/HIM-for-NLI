import torch

from torch import nn
from layers import BinaryTreeLSTM

class VariationalDropout(nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" (https://arxiv.org/abs/1506.02142) to a
    3D tensor.
    This module accepts a 3D tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
    and samples a single dropout mask of shape ``(batch_size, embedding_dim)`` and applies
    it to every time step.
    """
    def forward(self, input_tensor):
        """
        Apply dropout to input tensor.
        Parameters
        ----------
        input_tensor: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
        Returns
        -------
        output: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)`` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor

class EmbeddingLayer(nn.Module):
    """Implement embedding layer.
    """
    def __init__(self, vector_size, vocab_size, dropout=0.5):
        """
        Arguments:
            vector_size {int} -- word embedding size.
            vocab_size {int} -- vocabulary size.
        Keyword Arguments:
            dropout {float} -- dropout rate. (default: {0.5})
        """
        super(EmbeddingLayer, self).__init__()

        self.vector_size = vector_size
        self.embed = nn.Embedding(vocab_size, vector_size)
        self.dropout = VariationalDropout(dropout)

    def load(self, vectors):
        """Load pre-trained embedding weights.
        Arguments:
            vectors {torch.Tensor} -- from "TEXT.vocab.vectors".
        """
        self.embed.weight.data.copy_(vectors)

    def forward(self, x):
        """
        Arguments:
            x {torch.Tensor} -- input tensor with shape [batch_size, seq_length]
        """
        e = self.embed(x)
        return self.dropout(e)

class EncodingLayer(nn.Module):
    """BiLSTM encoder which encodes both the premise and hypothesis.
    """
    def __init__(self, input_size, hidden_size):
        super(EncodingLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=1,
                            bidirectional=True)

    def forward(self, x):
        """
        Arguments:
            x {torch.Tensor} -- input embeddings with shape [batch, seq_len, input_size]
        Returns:
            output {torch.Tensor} -- [batch, seq_len, num_directions * hidden_size]
        """
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        return output

class LocalInferenceModel(nn.Module):
    """The local inference model introduced in the paper.
    """
    def __init__(self):
        super(LocalInferenceModel, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

    def forward(self, p, h, p_mask, h_mask):
        """Apply local inference to premise and hyopthesis.
        Arguments:
            p {torch.Tensor} -- p has shape [batch, seq_len_p, 2 * hidden_size]
            h {torch.Tensor} -- h has shape [batch, seq_len_h, 2 * hidden_size]
            p_mask {torch.Tensor (int)} -- p has shape [batch, seq_len_p], 0 in the mask
                means padding.
            h_mask {torch.Tensor (int)} -- h has shape [batch, seq_len_h]
        Returns:
            m_p, m_h {torch.Tensor} -- tensor with shape [batch, seq_len, 8 * hidden_size]
        """
        # equation 11 in the paper:
        e = torch.matmul(p, h.transpose(1, 2))  # [batch, seq_len_p, seq_len_h]

        # masking the scores for padding tokens
        inference_mask = torch.matmul(p_mask.unsqueeze(2).float(),
                                      h_mask.unsqueeze(1).float())
        e.masked_fill_(inference_mask < 1e-7, -1e7)

        # equation 12 & 13 in the paper:
        h_score, p_score = self.softmax_1(e), self.softmax_2(e)
        h_ = h_score.transpose(1, 2).bmm(p)
        p_ = p_score.bmm(h)

        # equation 14 & 15 in the paper:
        m_p = torch.cat((p, p_, p - p_, p * p_), dim=-1)
        m_h = torch.cat((h, h_, h - h_, h * h_), dim=-1)

        assert inference_mask.shape == e.shape
        assert p.shape == p_.shape and h.shape == h_.shape
        assert m_p.shape[-1] == p.shape[-1] * 4

        return m_p, m_h


class CompositionLayer(nn.Module):
    """The composition layer.
    """
    def __init__(self, input_size, output_size, hidden_size, dropout=0.5):
        """
        Arguments:
            input_size {int} -- input size to the feedforward neural network.
            output_size {int} -- output size of the feedforward neural network.
            hidden_size {int} -- output hidden size of the LSTM model.
        Keyword Arguments:
            dropout {float} -- dropout rate (default: {0.5})
        """
        super(CompositionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.F = nn.Linear(input_size, output_size)
        self.lstm = nn.LSTM(output_size, hidden_size,
                            num_layers=1, bidirectional=True)
        self.dropout = VariationalDropout(dropout)

    def forward(self, m):
        """
        Arguments:
            m {torch.Tensor} -- [batch, seq_len, input_size]
        Returns:
            outputs {torch.Tensor} -- [batch, seq_len, hidden_size * 2]
        """
        y = self.dropout(self.F(m))
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(y)

        assert m.shape[:2] == outputs.shape[:2] and \
            outputs.shape[-1] == self.hidden_size * 2
        return outputs


class Pooling(nn.Module):
    """Apply maxing pooling and average pooling to the outputs of LSTM.
    """
    def __init__(self):
        super(Pooling, self).__init__()

    def forward(self, x, x_mask):
        """
        Arguments:
            x {torch.Tensor} -- [batch, seq_len, hidden_size * 2]
            x_mask {torch.Tensor} -- [batch, seq_len], 0 in the mask means padding
        Returns:
            v {torch.Tensor} -- [batch, hidden_size * 4]
        """
        mask_expand = x_mask.unsqueeze(-1).expand(x.shape)

        # average pooling
        x_ = x * mask_expand.float()
        v_avg = x_.sum(1) / x_mask.sum(-1).unsqueeze(-1).float()

        # max pooling
        x_ = x.masked_fill(mask_expand == 0, -1e7)
        v_max = x_.max(1).values

        assert v_avg.shape == v_max.shape == (x.shape[0], x.shape[-1])
        return torch.cat((v_avg, v_max), dim=-1)

class InferenceComposition(nn.Module):
    """Inference composition described in paper section 3.3
    """
    def __init__(self, input_size, output_size, hidden_size, dropout=0.5):
        """
        Arguments:
            input_size {int} -- input size to the feedforward neural network.
            output_size {int} -- output size of the feedforward neural network.
            hidden_size {int} -- output hidden size of the LSTM model.
        Keyword Arguments:
            dropout {float} -- dropout rate (default: {0.5})
        """
        super(InferenceComposition, self).__init__()
        self.composition = CompositionLayer(input_size,
                                            output_size,
                                            hidden_size,
                                            dropout=dropout)
        # self.composition_h = deepcopy(self.composition_p)
        self.pooling = Pooling()

    def forward(self, m_p, m_h, p_mask, h_mask):
        """
        Arguments:
            m_p {torch.Tensor} -- [batch, seq_len, input_size]
            m_h {torch.Tensor} -- [batch, seq_len, input_size]
            mask {torch.Tensor} -- [batch, seq_len], 0 means padding
        Returns:
            v {torch.Tensor} -- [batch, input_size * 8]
        """
        # equation 16 & 17 in the paper
        v_p, v_h = self.composition(m_p), self.composition(m_h)
        # equation 18 & 19 in the paper
        v_p_, v_h_ = self.pooling(v_p, p_mask), self.pooling(v_h, h_mask)
        # equation 20 in the paper
        v = torch.cat((v_p_, v_h_), dim=-1)

        assert v.shape == (m_p.shape[0], v_p.shape[-1] * 4)
        return v

class LinearSoftmax(nn.Module):
    """Implement the final linear layer.
    """
    def __init__(self, input_size, output_size, class_num, activation='relu', dropout=0.5):
        super(LinearSoftmax, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unknown activation function!!!")
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            self.dropout,
            nn.Linear(input_size, output_size),
            self.activation,
            # self.dropout,
            nn.Linear(output_size, class_num)
        )

    def forward(self, x):
        """
        Arguments:
            x {torch.Tensor} -- [batch, features]
        Returns:
            logits {torch.Tensor} -- raw, unnormalized scores for each class. [batch, class_num]
        """
        logits = self.mlp(x)
        return logits


class HIM(nn.Module):
    """Implement HIM model using the modules defined above.
    """

    def __init__(self,
                 hidden_size,
                 vector_size=64,
                 vocab_size = 1973,
                 num_labels=3,
                 dropout=0.5,
                 device='gpu'
                 ):
        """
        Arguments:
            vector_size {int} -- word embedding size.
            vocab_size {int} -- the size of the vocabulary.
            hidden_size {int} -- LSTM hidden size.

        Keyword Arguments:
            class_num {int} -- number of class for classification (default: {3})
            dropout {float} -- dropout rate (default: {0.5})
        """
        super(HIM, self).__init__()
        self.device = device
        self.embedding_layer = EmbeddingLayer(vector_size, vocab_size, dropout)
        self.seq_encoder = EncodingLayer(vector_size, hidden_size)
        self.tree_encoder = BinaryTreeLSTM(vector_size, hidden_size)
        self.inference = LocalInferenceModel()
        self.inferComp = InferenceComposition(hidden_size * 8,
                                              hidden_size,
                                              hidden_size,
                                              dropout)
        self.linear = LinearSoftmax(hidden_size * 16,
                                    hidden_size,
                                    num_labels,
                                    activation='tanh')

    def load_embeddings(self, vectors):
        """Load pre-trained word embeddings."""
        self.embedding_layer.load(vectors)

    def forward(self, p, p_length, h, h_length, p_word_idx, p_left, p_right, p_is_leaf, p_tree_len, h_word_idx, h_left, h_right, h_is_leaf, h_tree_len):
        """
        Arguments:
            p {torch.Tensor} -- premise [batch, seq_len]
            h {torch.Tensor} -- hypothesis [batch, seq_len]
            p_word_idx, etc. -- tree data
        Returns:
            logits {torch.Tensor} -- raw, unnormalized scores for each class
                with shape [batch, class_num]
        """
        # input embedding
        p_embed = self.embedding_layer(p)
        h_embed = self.embedding_layer(h)

        # sequential encoding
        p_seq = self.seq_encoder(p_embed)
        h_seq = self.seq_encoder(h_embed)

        p_mask = (p != 0).long()
        h_mask = (h != 0).long()

        m_p_seq, m_h_seq = self.inference(p_seq, h_seq, p_mask, h_mask)
        v_seq = self.inferComp(m_p_seq, m_h_seq, p_mask, h_mask)

        # tree encoding
        p_tree_h = self.tree_encoder(p_word_idx, p_left, p_right, p_is_leaf, self.embedding_layer.embed)
        h_tree_h = self.tree_encoder(h_word_idx, h_left, h_right, h_is_leaf, self.embedding_layer.embed)

        batch_size = p.shape[0]
        p_tree_seq_list = []
        h_tree_seq_list = []
        max_p_len = 0
        max_h_len = 0

        for b in range(batch_size):
            p_num_leaves = p_is_leaf[b, :p_tree_len[b]].sum().item()
            h_num_leaves = h_is_leaf[b, :h_tree_len[b]].sum().item()
            p_tree_seq_list.append(p_tree_h[b, :p_num_leaves])
            h_tree_seq_list.append(h_tree_h[b, :h_num_leaves])
            max_p_len = max(max_p_len, p_num_leaves)
            max_h_len = max(max_h_len, h_num_leaves)

        p_tree_seq = torch.zeros(batch_size, max_p_len, self.tree_encoder.hidden_size, device=p.device)
        h_tree_seq = torch.zeros(batch_size, max_h_len, self.tree_encoder.hidden_size, device=p.device)
        p_tree_mask = torch.zeros(batch_size, max_p_len, dtype=torch.long, device=p.device)
        h_tree_mask = torch.zeros(batch_size, max_h_len, dtype=torch.long, device=p.device)

        for b in range(batch_size):
            p_seq = p_tree_seq_list[b]
            h_seq = h_tree_seq_list[b]
            p_tree_seq[b, :len(p_seq)] = p_seq
            h_tree_seq[b, :len(h_seq)] = h_seq
            p_tree_mask[b, :len(p_seq)] = 1
            h_tree_mask[b, :len(h_seq)] = 1

        # make 2*hidden for compatibility
        p_tree_seq = torch.cat([p_tree_seq, p_tree_seq], dim=-1)
        h_tree_seq = torch.cat([h_tree_seq, h_tree_seq], dim=-1)

        m_p_tree, m_h_tree = self.inference(p_tree_seq, h_tree_seq, p_tree_mask, h_tree_mask)
        v_tree = self.inferComp(m_p_tree, m_h_tree, p_tree_mask, h_tree_mask)

        v = torch.cat((v_seq, v_tree), dim=-1)
        logits = self.linear(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


class ESIM(nn.Module):
    """Implement ESIM model using the modules defined above.
    """

    def __init__(self,
                 hidden_size,
                 vector_size=64,
                 vocab_size = 1973,
                 num_labels=3,
                 dropout=0.5,
                 device='gpu'
                 ):
        """
        Arguments:
            vector_size {int} -- word embedding size.
            vocab_size {int} -- the size of the vocabulary.
            hidden_size {int} -- LSTM hidden size.

        Keyword Arguments:
            class_num {int} -- number of class for classification (default: {3})
            dropout {float} -- dropout rate (default: {0.5})
        """
        super(ESIM, self).__init__()
        self.device = device
        self.embedding_layer = EmbeddingLayer(vector_size, vocab_size, dropout)
        self.encoder = EncodingLayer(vector_size, hidden_size)
        # self.hypo_encoder = deepcopy(self.premise_encoder)
        self.inference = LocalInferenceModel()
        self.inferComp = InferenceComposition(hidden_size * 8,
                                              hidden_size,
                                              hidden_size,
                                              dropout)
        self.linear = LinearSoftmax(hidden_size * 8,
                                    hidden_size,
                                    num_labels,
                                    activation='tanh')

    def load_embeddings(self, vectors):
        """Load pre-trained word embeddings.
        Arguments:
            vectors {torch.Tensor} -- pre-trained vectors.
        """
        self.embedding_layer.load(vectors)

    def forward(self, p, p_length, h, h_length):
        """
        Arguments:
            p {torch.Tensor} -- premise [batch, seq_len]
            h {torch.Tensor} -- hypothesis [batch, seq_len]
        Returns:
            logits {torch.Tensor} -- raw, unnormalized scores for each class
                with shape [batch, class_num]
        """
        # input embedding
        p_embeded = self.embedding_layer(p)
        h_embeded = self.embedding_layer(h)

        p_ = self.encoder(p_embeded)
        h_ = self.encoder(h_embeded)

        # local inference
        p_mask, h_mask = (p != 0).long(), (h != 0).long()
        m_p, m_h = self.inference(p_, h_, p_mask, h_mask)

        # inference composition
        v = self.inferComp(m_p, m_h, p_mask, h_mask)

        # final multi-layer perceptron
        logits = self.linear(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits,probabilities


