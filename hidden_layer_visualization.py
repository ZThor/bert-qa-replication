import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def run(model_name, question, context):
    # TODO remove padding tokens like [SEP]
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
    input_ids = tokenizer.encode(question, context)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # Segment A is the question, segment B is the answer?
    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # Output is a tuple with torch tensors
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    start_scores = output.start_logits
    end_scores = output.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = ' '.join(tokens[answer_start:answer_end + 1])

    hidden_states = output.hidden_states

    for i, layer in enumerate(hidden_states):
        # Token embedding of one of the 118 tokens in the question text
        hidden_layer = layer.squeeze(0)
        hidden_layer_np = hidden_layer.cpu().detach().numpy()

        reduction = PCA(n_components=2)
        reduced_layer = reduction.fit_transform(hidden_layer_np)
        plot_hidden_states(reduced_layer, tokens, question, answer, i)


def plot_hidden_states(reduced_layer, tokens, question, answer, layer_index):
    # TODO Supporting facts tokens
    # TODO differentiate if token was in question or context (based on where it was -> string embedding first question then context)
    x_data = [x[0] for x in reduced_layer]
    y_data = [x[1] for x in reduced_layer]
    # iterate over all datapoint
    for i in range(len(x_data)):
        color = "gray"
        marker = "o"
        if tokens[i] == answer:
            color = "red"
            marker = "d"
        elif tokens[i] in question:
            color = "orange"
            marker = "*"
        plt.scatter(x_data[i], y_data[i], c=color, marker=marker)
        plt.text(x_data[i] + 0.1, y_data[i] + 0.2, tokens[i], fontsize=6)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    filename = "layer" + str(layer_index) + ".pdf"
    plt.savefig(filename)
    plt.clf()


if __name__ == '__main__':
    question = "What is a common punishment in the UK and Ireland?"

    context = "Currently detention is one of the most common punishments in schools in the United States, the UK, Ireland, " \
              "Singapore and other countries. It requires the pupil to remain in school at a given time in the school day " \
              "(such as lunch, recess or after school); or even to attend school on a non-school day, e.g. \"Saturday detention\" " \
              "held at some schools. During detention,students normally have to sit in a classroom and do work, write lines or" \
              " a punishment essay, or sit quietly."

    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    run(model_name, question, context)