from .base import Tokenizer
from .basic import BasicTokenizer
from .regex import RegexTokenizer

try:
    from .gpt4 import GPT4Tokenizer
except ImportError:
    # tiktoken not installed; GPT4Tokenizer unavailable
    GPT4Tokenizer = None
