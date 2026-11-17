"""
Autonomous Invariant Task Generator with Scaffolding Skeletons.
Generates structured programming challenges with explicit function signatures
and edge-case assertions spanning strings, math, arrays, and dictionaries.
"""

from typing import Dict, Any, List

TASK_TEMPLATES = [
    {
        "name": "Safe Number Clamping",
        "category": "math",
        "entry_point": "clamp_value",
        "skeleton": "def clamp_value(val: float, low: float, high: float) -> float:",
        "prompt": "Write a function `clamp_value(val, low, high)` that restricts val between low and high. If low > high, swap them. Return clamped float/int.",
        "tests": """
assert clamp_value(5, 0, 10) == 5
assert clamp_value(-5, 0, 10) == 0
assert clamp_value(15, 0, 10) == 10
assert clamp_value(5, 10, 0) == 5
assert clamp_value(0, 0, 0) == 0
"""
    },
    {
        "name": "Safe Zero Division Guard",
        "category": "math",
        "entry_point": "safe_divide",
        "skeleton": "def safe_divide(a: float, b: float, default: float = 0.0) -> float:",
        "prompt": "Write a function `safe_divide(a, b, default=0.0)` that returns a / b. If b == 0 or either is None, return default.",
        "tests": """
assert safe_divide(10, 2) == 5.0
assert safe_divide(10, 0) == 0.0
assert safe_divide(10, 0, default=-1.0) == -1.0
assert safe_divide(None, 5) == 0.0
assert safe_divide(5, None) == 0.0
"""
    },
    {
        "name": "Palindrome Invariant Filter",
        "category": "string",
        "entry_point": "is_palindrome",
        "skeleton": "def is_palindrome(s: str) -> bool:",
        "prompt": "Write a function `is_palindrome(s)` that checks if string s is a palindrome, ignoring non-alphanumeric characters and case. Empty string returns True.",
        "tests": """
assert is_palindrome("A man, a plan, a canal: Panama") is True
assert is_palindrome("race a car") is False
assert is_palindrome("") is True
assert is_palindrome(".,") is True
assert is_palindrome("0P") is False
"""
    },
    {
        "name": "Count Element Frequency",
        "category": "array",
        "entry_point": "count_frequency",
        "skeleton": "def count_frequency(lst: list, target: Any) -> int:",
        "prompt": "Write a function `count_frequency(lst, target)` that returns the count of target in lst. If lst is empty or None, return 0.",
        "tests": """
assert count_frequency([1, 2, 2, 3, 2], 2) == 3
assert count_frequency([], 1) == 0
assert count_frequency(None, 1) == 0
assert count_frequency(["a", "b", "a"], "a") == 2
"""
    },
    {
        "name": "Reverse Vowels Only",
        "category": "string",
        "entry_point": "reverse_vowels",
        "skeleton": "def reverse_vowels(s: str) -> str:",
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
        "entry_point": "flatten_nested",
        "skeleton": "def flatten_nested(lst: list) -> list:",
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
        "entry_point": "group_anagrams",
        "skeleton": "def group_anagrams(words: list) -> list:",
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
        "entry_point": "prime_factors",
        "skeleton": "def prime_factors(n: int) -> list:",
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
        "name": "Run Length Encoding",
        "category": "string",
        "entry_point": "rle_encode",
        "skeleton": "def rle_encode(s: str) -> str:",
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
        "entry_point": "deep_merge",
        "skeleton": "def deep_merge(d1: dict, d2: dict) -> dict:",
        "prompt": "Write a Python function `deep_merge(d1, d2)` that recursively merges d2 into d1. If values for same key are both dicts, merge recursively; otherwise d2 overwrites d1. Return new dict.",
        "tests": """
d1 = {'a': 1, 'b': {'x': 10, 'y': 20}}
d2 = {'b': {'y': 30, 'z': 40}, 'c': 3}
res = deep_merge(d1, d2)
assert res == {'a': 1, 'b': {'x': 10, 'y': 30, 'z': 40}, 'c': 3}
assert deep_merge({}, {'a': 1}) == {'a': 1}
assert deep_merge({'a': 1}, {}) == {'a': 1}
"""
    }
]


class InvariantTaskGenerator:
    """Produces endless batches of structured programming challenges with edge-case tests."""

    def __init__(self):
        self.history_index = 0

    def sample_batch(self, batch_size: int = 2) -> List[Dict[str, Any]]:
        """Sample a batch of tasks, cycling with variations."""
        batch = []
        for _ in range(batch_size):
            template = TASK_TEMPLATES[self.history_index % len(TASK_TEMPLATES)]
            self.history_index += 1
            batch.append({
                "id": f"task_{self.history_index}_{template['category']}",
                "name": template["name"],
                "category": template["category"],
                "entry_point": template["entry_point"],
                "skeleton": template["skeleton"],
                "prompt": template["prompt"],
                "tests": template["tests"],
            })
        return batch
