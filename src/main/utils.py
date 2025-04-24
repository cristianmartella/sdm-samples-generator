from functools import reduce
import operator
import re
import random
import nltk
from nltk.corpus import wordnet

# Matcher types
MATCHER_TYPE_SENTENCE = "sentence"


# Matcher patterns
"""
Matches all sentences included in "..." that do not start with a number and a dash and do not contain a single word.
"""
PATTERN_SENTENCE = r"^(?!\w+\-\w+|\w+$|\w+\:(\w|\/)+|\d+.\d+)\w+(\W|\S)+$"

sentence_value_regex = re.compile(PATTERN_SENTENCE)





def search_dict(item, regex, prefix='') -> list:
    """
    Recursively search for a value in a dict item.
    :param item: The dict item to search in.
    :param regex: The regex pattern to search for.
    :param prefix: The prefix to add to the key.
    :return: A list of keys that match the regex pattern.
    """
    result = []
    if isinstance(item, dict):
        for key, value in item.items():
            pref = "{}.{}".format(prefix, key)
            result.extend(search_dict(value, regex, pref))
    elif isinstance(item, list):
        for index, value in enumerate(item):
            pref = "{}.{}".format(prefix, index)
            result.extend(search_dict(value, regex, pref))
    elif isinstance(item, (str, int, float)) and re.search(regex, str(item)):
        result.append(prefix[1:])
   	
    #print(f"search_dict({item}, {regex}, {prefix}) ---> result: {result}")
    return result



# matcher function
def _matcher(data, value_regex, key_regex=None):
    """
    Core matcher function.
    :param data: The data to search in.
    :param value_regex: The value regex pattern to search for.
    :param key_regex: The kye regex pattern to search for.
    :return: A list of key/values paths to the matching regex.
    """

    # search for value paths in data
    value_paths = search_dict(data, value_regex)

    # search for matching keys within value paths (key_regex for finer search)
    key_paths = []
    if key_regex:
        for path in value_paths:
            path_list = path.split('.')
            matching_keys = list(filter(lambda key: key_regex.search(str(key)), path_list))
            if matching_keys:
                key_paths.append(path)

    return key_paths if key_regex else value_paths




def match(type, data):
    """
    Matcher wrapper function.
    :param type: The type of matcher.
    :param data: The data to search in.
    :return: A list of key paths that match the type.
    """
    if type == MATCHER_TYPE_SENTENCE:
        return _matcher(data, sentence_value_regex)


def recursive_get(d:dict, keys:list) -> dict:
    """
    Recursively get a value from a dictionary using a list of keys.
    :param d: The dictionary to search.
    :param keys: The list of keys to search for.
    :return: The value found at the given keys.
    """
    #return reduce(lambda c, k: c.get(k, {}) if isinstance(c, dict) else c[int(k)], keys, d)
    return reduce(operator.getitem, keys, d)


def clear_properties(data:dict, propertyPaths:list) -> dict:
    """
    Clear properties from a dictionary.
    :param data: The dictionary to clear properties from.
    :param propertyPaths: The list of property paths to clear.
    :return: The modified dictionary.
    """
    for path in propertyPaths:
        keys = path.split('.')
        try:
            # Normalized format
            recursive_get(data, keys[:-1])[keys[-1]] = ""
        except TypeError as e:
            # Key-value format
            recursive_get(data, keys[:-2])[keys[-2]] = ""
        except KeyError as e:
            pass
        
    return data



def get_random_synonym(word:str) -> str:
    """
    Get a random synonym for a word.
    :param word: The word to get a synonym for.
    :return: A random synonym for the word.
    """
    sim_threshold = 0.1
    synonyms = set()

    # Use Wordnet Path Similarity to keep most similar matches (above a threshold)
    synonyms_synsets = wordnet.synsets(word)
    if len(synonyms_synsets) > 0:
        word_synset = synonyms_synsets[0]

    synonyms_synsets_filtered = [synset for synset in synonyms_synsets if synset.path_similarity(word_synset) > sim_threshold]

    print(f"word: {word}")
    # print(f"synonyms_synsets: {synonyms_synsets}")
    # print(f"synonyms_synsets_filtered: {synonyms_synsets_filtered}")

    if len(synonyms_synsets_filtered) > 0:
        for syn in synonyms_synsets_filtered:
            syn_name = syn.lemmas()[0].name()
            #print(f"syn name: {syn_name}")
            synonyms.add(''.join(syn_name))

        print(f"synonyms: {synonyms}")
        # Return a random synonym
        return random.choice(list(synonyms))
    else:
        return word


def randomize_camel_word(word:str) -> str:
    """
    Randomize a camel case word.
    :param word: The camel case word to randomize.
    :return: The randomized camel case word.
    """
    # Split the camel case word into a list of words
    words = camel_case_split(word)

    # Get a random synonym for each word in the camel case word
    for word in words:
        # Get a random synonym for the word
        synonym = get_random_synonym(word)
        # Replace the word with the synonym
        words[words.index(word)] = synonym.title()

    # Join the words back together
    randomizedWord = ''.join(words).replace("_", "")
    return randomizedWord[0].lower() + randomizedWord[1:]


def camel_case_split(str:str) -> list:
    """
    Split a camel case string into words.
    :param str: The camel case string to split.
    :return: A list of words.
    """
    return re.findall(r'[A-Za-z-](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str)


def camel_to_snake(word:str) -> str:
    """
    Convert a camel case word to snake case.
    :param word: The camel case word to convert.
    :return: The snake case word.
    """
    word = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', word)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', word).lower()


def dict_to_snake_keys(dd:dict) -> dict:
    """
    Convert the keys of a dictionary to snake case.
    :param dd: The dictionary to convert.
    :return: The converted dictionary.
    """
    if isinstance(dd, dict):
        return {camel_to_snake(k): dict_to_snake_keys(v) for k, v in dd.items()}
    elif isinstance(dd, list):
        return [dict_to_snake_keys(x) for x in dd]
    else:
        return dd