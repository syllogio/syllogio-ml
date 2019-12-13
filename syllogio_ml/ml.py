import tensorflow
from tensorflow import keras
import spacy
import numpy as np
import itertools
import csv
import pprint
pp = pprint.PrettyPrinter(indent=2)

# Configuration.
spacy_model_name = "en_core_web_sm"
training_data_path = "syllogio_ml/training_data.csv"
max_syllogism_proposition_count = 20
max_proposition_token_data_count = 2000

# Initialize spacy models on training data.
nlp = spacy.load(spacy_model_name)


# Flattens an input array.
def flatten(input):
    return list(itertools.chain(*input))


# Returns true if given "true" and false if given "false".
def bool_to_int(bool):
    return int(bool == "true")


# Pads a given list so that it's length is len(max).
def pad(input, max, padWith=0):
    return input + [padWith] * (max - len(input))


# Fetch training data and parse.
def get_train_data_list(filepath):
    data = []
    labels = []
    with open(filepath, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter="|")
        for row in reader:
            [permutations, expectations] = create_permutations_for_syllogism(
                row, row[0]
            )
            data.append(permutations)
            labels.append(expectations)

    return tuple([flatten(data), flatten(labels)])


# Create all permutations for a given input.
def create_permutations_for_syllogism(propositions, conclusion):
    permutations = []
    labels = []
    propLen = len(propositions)
    for comb in itertools.permutations(propositions, propLen):
        conclusionIndex = comb.index(conclusion)
        answer = [
            1 if i == conclusionIndex else 0
            for i in range(propLen)
        ]
        permutations.append(list(comb))
        labels.append(answer)

    return [permutations, labels]


def token_to_train_data(token):
    return [
        token.i,
        token.dep,
        token.pos,
        token.orth,
        token.lemma,
        token.norm,
        token.tag,
        bool_to_int(token.is_alpha),
        bool_to_int(token.is_ascii),
        bool_to_int(token.is_bracket),
        bool_to_int(token.is_currency),
        bool_to_int(token.is_digit),
        bool_to_int(token.is_left_punct),
        bool_to_int(token.is_right_punct),
        bool_to_int(token.is_quote),
        bool_to_int(token.is_quote),
        bool_to_int(token.is_sent_start),
        bool_to_int(token.is_space),
        bool_to_int(token.is_stop),
        bool_to_int(token.is_stop),
    ]


def proposition_to_train_data(proposition):
    proposition_data = []
    for token in nlp(proposition):
        token_data = token_to_train_data(token) + token_to_train_data(token.head)
        proposition_data.append(token_data)

    return pad(flatten(proposition_data), max_proposition_token_data_count)


def format_input_data(syllogisms):
    data = []
    for syllogism in syllogisms:
        sdata = []
        for proposition in syllogism:
            sdata.append(proposition_to_train_data(proposition))
        data.append(
            pad(
                sdata,
                max_syllogism_proposition_count,
                pad([], max_proposition_token_data_count),
            )
        )

    return tensorflow.convert_to_tensor(np.asarray(data, dtype=float), dtype=float)


def format_input_labels(labels):
    data = []
    for label in labels:
        data.append(pad([float(i) for i in label], max_syllogism_proposition_count))
    return tensorflow.convert_to_tensor(np.asarray(data, dtype=float), dtype=float)


[train_syllogisms, train_labels] = get_train_data_list(training_data_path)
pp.pprint([train_syllogisms, train_labels])

test_syllogisms = [
    ["All men are mortal", "Socrates is mortal", "Socrates is a man"],
    ["China has a GDP", "All countries have a GDP", "China is a country"],
    [
        "Garbage is rotting material",
        "Humans do not like rotting material",
        "Humans do not like garbage",
    ],
]
train_data = format_input_data(train_syllogisms)
test_data = format_input_data(test_syllogisms)
train_expectations = format_input_labels(train_labels)

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(max_syllogism_proposition_count, max_proposition_token_data_count)),
        keras.layers.Dense(max_syllogism_proposition_count, activation="relu"),
        keras.layers.Dense(max_syllogism_proposition_count, activation="relu"),
        keras.layers.Dense(max_syllogism_proposition_count, activation="relu"),
        keras.layers.Dense(max_syllogism_proposition_count, activation="relu"),
        keras.layers.Dense(max_syllogism_proposition_count, activation="relu"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

model.fit(train_data, train_expectations, epochs=100)

predictions = model.predict(test_data)

for index, prediction in enumerate(predictions):
    pp.pprint(prediction)
    predictionIndex = np.argmax(prediction)
    if predictionIndex >= len(test_syllogisms[index]):
        print("Well, this prediction was way off base")
    else:
        print(test_syllogisms[index][predictionIndex])

