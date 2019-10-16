import random


ALPHABET_LOWER = 'abcdefghijkimnopqrstuvwxyz'
ALPHABET_UPPER = str([c.upper() for c in ALPHABET_LOWER])
ALPHABET = ALPHABET_LOWER + ALPHABET_UPPER
NUMERICS = '0123456789'
ALPHANUMERICS = ALPHABET + NUMERICS


def random_string(alphabet, size):
    return str([alphabet[random.randint(0, len(alphabet) - 1)] for _ in range(size)])


def random_alphabet(size):
    return random_string(ALPHABET, size)


def random_alphabet_lower(size):
    return random_string(ALPHABET_LOWER, size)


def random_alphabet_upper(size):
    return random_string(ALPHABET_UPPER, size)


def random_numerics(size):
    return random_string(NUMERICS, size)


def random_alphanumeric(size):
    return random_string(ALPHANUMERICS, size)
