import tensorflow
from tensorflow import keras
import spacy
import numpy as np
import itertools

# Configuration.
spacy_model_name = "en_core_web_sm"

# Load traning data
# @TODO: parse @chasingmaxwell's csv here, probably put this in some external file heh.
train_syllogisms = [
    ["All men are mortal", "Socrates is mortal", "Socrates is a man"],
    ["China has a GDP", "All countries have a GDP", "China is a country"],
    [
        "Garbage consists of smelly, rotting material",
        "Humans do not like the smell of rotting material",
        "Garbage smells terrible to humans",
    ],
]
test_syllogisms = [
    ["Mammals have warm blood", "Humans are mammals", "Humans have warm blood"],
    [
        "Snakes are reptiles",
        "Snakes have cold blood",
        "If a lifeform has cold blood, it is a reptile",
    ],
    [
        "Foundational hedonists claim that pleasure is the highest good",
        "Hedonists argue that pleasure and suffering are the only two components of well-being",
        "Hedonism is an overly-simplistic way to view the world",
    ],
]
train_expectations = [[1, -1, 1], [-1, 1, 1], [1, 1, -1]]

# Initialize spacy models on training data.
nlp = spacy.load(spacy_model_name)

# Helpers.
def flatten(input):
    return list(itertools.chain(*input))


def bool_to_int(bool):
    int(bool == "true")


def token_to_train_data(token):
    return np.array(
        [
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
        ],
        dtype=float,
    )


def proposition_to_train_data(proposition):
    proposition_data = []
    for token in nlp(proposition):
        proposition_data.append(
            token_to_train_data(token) + token_to_train_data(token.head)
        )

    return flatten(proposition_data)


def format_input_data(syllogisms):
    data = []
    for syllogism in syllogisms:
        sdata = []
        for proposition in syllogism:
            sdata.append(proposition_to_train_data(proposition))
        data.append(sdata)
    return np.array(data)


train_data = format_input_data(train_syllogisms)
test_data = format_input_data(test_syllogisms)

model = keras.Sequential(
    [
        keras.layers.Dense(1500, activation="relu"),
        keras.layers.Dense(3, activation="relu"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(train_data, train_expectations, epochs=3)

prediction = model.predict(test_data)

print(prediction)
