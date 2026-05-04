from .arc import AI2ArcTask
from .base import FewShotTask, Task
from .gsm8k import Gsm8kTask
from .hellaswag import HellaSwagTask
from .math import MathTask
from .mbpp2 import Mbpp2Task
from .mmlu import MMLUTask

try:
    from .cls import ClsTask
except ImportError:
    pass
