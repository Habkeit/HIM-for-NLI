"""
Preprocessor and dataset definition for NLI.
"""

import string
import torch
import numpy as np
from nltk import tree

from collections import Counter
from torch.utils.data import Dataset


class Preprocessor(object):
    """
    Preprocessor class for Natural Language Inference datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 labeldict={},
                 bos=None,
                 eos=None):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict
        self.bos = bos
        self.eos = eos

    def read_data(self, filepath):
        """
        Read the premises, hypotheses and labels from some NLI dataset's
        file and return them in a dictionary. The file should be in the same
        form as SNLI's .txt files.

        Args:
            filepath: The path to a file containing some premises, hypotheses
                and labels that must be read. The file should be formatted in
                the same way as the SNLI (and MultiNLI) dataset.

        Returns:
            A dictionary containing three lists, one for the premises, one for
            the hypotheses, and one for the labels in the input data.
        """
        with open(filepath, "r", encoding="utf8") as input_data:
            ids, premises, hypotheses, labels = [], [], [], []
            premises_parse, hypotheses_parse = [], []

            # Translation tables to remove parentheses and punctuation from
            # strings.
            parentheses_table = str.maketrans({"(": None, ")": None})
            punct_table = str.maketrans({key: " "
                                         for key in string.punctuation})

            # Ignore the headers on the first line of the file.
            next(input_data)

            for line in input_data:
                line = line.strip().split("\t")

                # Ignore sentences that have no gold label.
                if line[0] == "-":
                    continue

                pair_id = line[7]
                premise = line[1]
                hypothesis = line[2]
                premise_parse = line[1]  # binary parse
                hypothesis_parse = line[2]  # binary parse

                # Remove '(' and ')' from the premises and hypotheses.
                premise = premise.translate(parentheses_table)
                hypothesis = hypothesis.translate(parentheses_table)

                if self.lowercase:
                    premise = premise.lower()
                    hypothesis = hypothesis.lower()

                if self.ignore_punctuation:
                    premise = premise.translate(punct_table)
                    hypothesis = hypothesis.translate(punct_table)

                # Each premise and hypothesis is split into a list of words.
                premises.append([w for w in premise.rstrip().split()
                                 if w not in self.stopwords])
                hypotheses.append([w for w in hypothesis.rstrip().split()
                                   if w not in self.stopwords])
                premises_parse.append(premise_parse)
                hypotheses_parse.append(hypothesis_parse)
                labels.append(line[0])
                ids.append(pair_id)

            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "premises_parse": premises_parse,
                    "hypotheses_parse": hypotheses_parse,
                    "labels": labels}

    def build_worddict(self, data):
        """
        Build a dictionary associating words to unique integer indices for
        some dataset. The worddict can then be used to transform the words
        in datasets to their indices.

        Args:
            data: A dictionary containing the premises, hypotheses and
                labels of some NLI dataset, in the format returned by the
                'read_data' method of the Preprocessor class.
        """
        words = []
        [words.extend(sentence) for sentence in data["premises"]]
        [words.extend(sentence) for sentence in data["hypotheses"]]

        counts = Counter(words)
        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)

        self.worddict = {}

        # Special indices are used for padding, out-of-vocabulary words, and
        # beginning and end of sentence tokens.
        self.worddict["_PAD_"] = 0
        self.worddict["_OOV_"] = 1

        offset = 2
        if self.bos:
            self.worddict["_BOS_"] = 2
            offset += 1
        if self.eos:
            self.worddict["_EOS_"] = 3
            offset += 1

        for i, word in enumerate(counts.most_common(num_words)):
            self.worddict[word[0]] = i + offset

        if self.labeldict == {}:
            label_names = set(data["labels"])
            self.labeldict = {label_name: i
                              for i, label_name in enumerate(label_names)}

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict["_OOV_"]
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def indices_to_words(self, indices):
        """
        Transform the indices in a list to their corresponding words in
        the object's worddict.

        Args:
            indices: A list of integer indices corresponding to words in
                the Preprocessor's worddict.

        Returns:
            A list of words.
        """
        return [list(self.worddict.keys())[list(self.worddict.values())
                                           .index(i)]
                for i in indices]

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"ids": [],
                            "premises": [],
                            "hypotheses": [],
                            "premises_parse": [],
                            "hypotheses_parse": [],
                            "labels": []}

        for i, premise in enumerate(data["premises"]):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue

            transformed_data["ids"].append(data["ids"][i])

            if label == "hidden":
                transformed_data["labels"].append(-1)
            else:
                transformed_data["labels"].append(self.labeldict[label])

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

            transformed_data["premises_parse"].append(data["premises_parse"][i])
            transformed_data["hypotheses_parse"].append(data["hypotheses_parse"][i])

        return transformed_data

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.

        Args:
            embeddings_file: A file containing pretrained word embeddings.

        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self.worddict.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)

        return embedding_matrix


class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {"ids": [],
                     "premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}

        for i, premise in enumerate(data["premises"]):
            self.data["ids"].append(data["ids"][i])
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

        # Process trees. Use parse strings when available, otherwise fall back
        # to a simple left-branching tree built from the sentence indices.
        self.premises_trees = []
        self.hypotheses_trees = []
        if "premises_parse" in data and "hypotheses_parse" in data:
            for i, parse_str in enumerate(data["premises_parse"]):
                tree_obj = tree.Tree.fromstring(parse_str)
                premise = data["premises"][i]  # list of indices
                flattened = self.flatten_tree(tree_obj, premise)
                self.premises_trees.append(flattened)
            for i, parse_str in enumerate(data["hypotheses_parse"]):
                tree_obj = tree.Tree.fromstring(parse_str)
                hypothesis = data["hypotheses"][i]
                flattened = self.flatten_tree(tree_obj, hypothesis)
                self.hypotheses_trees.append(flattened)
        else:
            for premise in data["premises"]:
                flattened = self.linear_fallback_tree(premise)
                self.premises_trees.append(flattened)
            for hypothesis in data["hypotheses"]:
                flattened = self.linear_fallback_tree(hypothesis)
                self.hypotheses_trees.append(flattened)

    def flatten_tree(self, tree_obj, sentence_indices):
        nodes = []
        leaf_idx = 0
        def traverse(node):
            nonlocal leaf_idx
            if isinstance(node, str):  # leaf
                word_idx = sentence_indices[leaf_idx]
                leaf_idx += 1
                nodes.append({'word_idx': word_idx, 'left': -1, 'right': -1, 'is_leaf': True})
                return len(nodes) - 1
            else:
                left_idx = traverse(node[0]) if len(node) > 0 else -1
                right_idx = traverse(node[1]) if len(node) > 1 else -1
                nodes.append({'word_idx': -1, 'left': left_idx, 'right': right_idx, 'is_leaf': False})
                return len(nodes) - 1
        traverse(tree_obj)
        return nodes

    def linear_fallback_tree(self, sentence_indices):
        nodes = []
        leaf_positions = []
        for idx in sentence_indices:
            nodes.append({'word_idx': idx, 'left': -1, 'right': -1, 'is_leaf': True})
            leaf_positions.append(len(nodes) - 1)

        # Build a simple left-branching tree over the leaves.
        while len(leaf_positions) > 1:
            left = leaf_positions.pop(0)
            right = leaf_positions.pop(0)
            nodes.append({'word_idx': -1, 'left': left, 'right': right, 'is_leaf': False})
            leaf_positions.insert(0, len(nodes) - 1)

        return nodes

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                "premise_tree": self.premises_trees[index],
                "hypothesis_tree": self.hypotheses_trees[index],
                "label": self.data["labels"][index]}


def collate_fn(batch):
    # batch is list of dicts
    ids = [item['id'] for item in batch]
    premises = torch.stack([item['premise'] for item in batch])
    premise_lengths = torch.tensor([item['premise_length'] for item in batch])
    hypotheses = torch.stack([item['hypothesis'] for item in batch])
    hypothesis_lengths = torch.tensor([item['hypothesis_length'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # for trees
    premise_trees = [item['premise_tree'] for item in batch]
    hypothesis_trees = [item['hypothesis_tree'] for item in batch]

    max_p_nodes = max(len(t) for t in premise_trees) if premise_trees else 0
    max_h_nodes = max(len(t) for t in hypothesis_trees) if hypothesis_trees else 0

    p_word_idx = torch.zeros(len(batch), max_p_nodes, dtype=torch.long)
    p_left = torch.full((len(batch), max_p_nodes), -1, dtype=torch.long)
    p_right = torch.full((len(batch), max_p_nodes), -1, dtype=torch.long)
    p_is_leaf = torch.zeros(len(batch), max_p_nodes, dtype=torch.bool)
    p_lengths = []

    for b, tree in enumerate(premise_trees):
        p_lengths.append(len(tree))
        for j, node in enumerate(tree):
            p_word_idx[b, j] = node['word_idx']
            p_left[b, j] = node['left']
            p_right[b, j] = node['right']
            p_is_leaf[b, j] = node['is_leaf']

    h_word_idx = torch.zeros(len(batch), max_h_nodes, dtype=torch.long)
    h_left = torch.full((len(batch), max_h_nodes), -1, dtype=torch.long)
    h_right = torch.full((len(batch), max_h_nodes), -1, dtype=torch.long)
    h_is_leaf = torch.zeros(len(batch), max_h_nodes, dtype=torch.bool)
    h_lengths = []

    for b, tree in enumerate(hypothesis_trees):
        h_lengths.append(len(tree))
        for j, node in enumerate(tree):
            h_word_idx[b, j] = node['word_idx']
            h_left[b, j] = node['left']
            h_right[b, j] = node['right']
            h_is_leaf[b, j] = node['is_leaf']

    return {
        'ids': ids,
        'premises': premises,
        'premise_lengths': premise_lengths,
        'hypotheses': hypotheses,
        'hypothesis_lengths': hypothesis_lengths,
        'labels': labels,
        'premise_word_idx': p_word_idx,
        'premise_left': p_left,
        'premise_right': p_right,
        'premise_is_leaf': p_is_leaf,
        'premise_tree_lengths': torch.tensor(p_lengths),
        'hypothesis_word_idx': h_word_idx,
        'hypothesis_left': h_left,
        'hypothesis_right': h_right,
        'hypothesis_is_leaf': h_is_leaf,
        'hypothesis_tree_lengths': torch.tensor(h_lengths)
    }
