import think
from think import Memory, Str

def test_memory_probability_methods():

    class Word(Memory, Str):
        pass

    sentences = [
        "the apple is red",
        "china is red",
        "apple computers",
        "green apple",
        "yellow apple",
        "the red guard in china",
    ]
    for sentence in sentences:
        words = sentence.split(' ')
        Word.connect(words)
        for w in words:
            Word(w).connect(words)

    assert Word('red').prob('red') == 1
    assert Word('apple').prob('apple') == 1
    assert Word('china').prob('china') == 1

    assert Word('red').prob('apple') == 1/3
    assert Word('apple').prob('red') == 1/4
    assert Word.prob_a_given_b('apple', 'red') == 1/3
    assert Word.prob_a_given_b('red', 'apple') == 1/4

    assert Word.total() == 18
    assert Word.count('apple') == 4
    assert Word('apple').total() == 10

