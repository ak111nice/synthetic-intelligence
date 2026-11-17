"""
Autonomous Invariant Task Generator.
Generates parameterized algorithmic problems with strict edge-case assertions
spanning strings, math, arrays, and dictionaries.
"""

import random
from typing import Dict, Any, List

TASK_TEMPLATES = [
    {
        "name": "Reverse Vowels Only",
        "category": "string",
        "prompt": "Write a Python function `reverse_vowels(s)` that reverses ONLY the vowels (a, e, i, o, u, case-insensitive) in string s, leaving consonants and symbols in their original positions. Return string.",
        "tests": """
assert reverse_vowels("hello") == "holle"
assert reverse_vowels("leetcode") == "leotcede"
assert reverse_vowels("") == ""
assert reverse_vowels("bcdfg") == "bcdfg"
assert reverse_vowels("aA") == "Aa"
assert reverse_vowels("a.b,.") == "a.b,."
"""
    },
    {
        "name": "Flatten Nested List",
        "category": "array",
        "prompt": "Write a Python function `flatten_nested(lst)` that flattens an arbitrarily deep nested list into a single flat list. Empty lists should be omitted. Return flat list.",
        "tests": """
assert flatten_nested([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]
assert flatten_nested([]) == []
assert flatten_nested([[[]]]) == []
assert flatten_nested([1, [2], [3, [4, [5]]]]) == [1, 2, 3, 4, 5]
"""
    },
    {
        "name": "Group Anagrams",
        "category": "dictionary",
        "prompt": "Write a Python function `group_anagrams(words)` that groups words with identical character frequencies together. Returns list of lists. Order within groups does not matter, but each group must be sorted.",
        "tests": """
res = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
res_sorted = sorted([sorted(g) for g in res])
assert res_sorted == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]
assert group_anagrams([]) == []
assert group_anagrams(["a"]) == [["a"]]
"""
    },
    {
        "name": "Prime Factors List",
        "category": "math",
        "prompt": "Write a Python function `prime_factors(n)` that returns a sorted list of all prime factors of integer n. If n <= 1, return empty list.",
        "tests": """
assert prime_factors(12) == [2, 2, 3]
assert prime_factors(1) == []
assert prime_factors(0) == []
assert prime_factors(17) == [17]
assert prime_factors(315) == [3, 3, 5, 7]
"""
    },
    {
        "name": "Matrix Transpose with Ragged Protection",
        "category": "array",
        "prompt": "Write a Python function `transpose_matrix(m)` that transposes a 2D matrix. If m is empty or inner rows have mismatched lengths, return None.",
        "tests": """
assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
assert transpose_matrix([[1]]) == [[1]]
assert transpose_matrix([]) is None
assert transpose_matrix([[]]) is None
assert transpose_matrix([[1, 2], [3]]) is None
"""
    },
    {
        "name": "Run Length Encoding",
        "category": "string",
        "prompt": "Write a Python function `rle_encode(s)` that encodes repeated characters into count + char pairs (e.g. 'AAB' -> '2A1B'). If s is empty, return empty string.",
        "tests": """
assert rle_encode("AAABBBCC") == "3A3B2C"
assert rle_encode("A") == "1A"
assert rle_encode("") == ""
assert rle_encode("ABBA") == "1A2B1A"
"""
    },
    {
        "name": "Deep Dictionary Merger",
        "category": "dictionary",
        "prompt": "Write a Python function `deep_merge(d1, d2)` that recursively merges d2 into d1. If values for same key are both dicts, merge recursively; otherwise d2 overwrites d1. Return new dict.",
        "tests": """
d1 = {'a': 1, 'b': {'x': 10, 'y': 20}}
d2 = {'b': {'y': 30, 'z': 40}, 'c': 3}
res = deep_merge(d1, d2)
assert res == {'a': 1, 'b': {'x': 10, 'y': 30, 'z': 40}, 'c': 3}
assert deep_merge({}, {'a': 1}) == {'a': 1}
assert deep_merge({'a': 1}, {}) == {'a': 1}
"""
    },
    {
        "name": "Roman Numeral Decoder",
        "category": "math",
        "prompt": "Write a Python function `roman_to_int(s)` that converts Roman numeral string s ('I', 'V', 'X', 'L', 'C', 'D', 'M') to integer. Case insensitive. If invalid character, return -1.",
        "tests": """
assert roman_to_int("III") == 3
assert roman_to_int("IV") == 4
assert roman_to_int("IX") == 9
assert roman_to_int("LVIII") == 58
assert roman_to_int("MCMXCIV") == 1994
assert roman_to_int("invalid") == -1
assert roman_to_int("") == 0
"""
    }
]


class InvariantTaskGenerator:
    """Produces endless batches of structured programming challenges with edge-case tests."""

    def __init__(self):
        self.history_index = 0

    def sample_batch(self, batch_size: int = 3) -> List[Dict[str, Any]]:
        """Sample a batch of tasks, cycling with variations."""
        batch = []
        for _ in range(batch_size):
            template = TASK_TEMPLATES[self.history_index % len(TASK_TEMPLATES)]
            self.history_index += 1
            batch.append({
                "id": f"task_{self.history_index}_{template['category']}",
                "name": template["name"],
                "category": template["category"],
                "prompt": template["prompt"],
                "tests": template["tests"],
            })
        return batch
