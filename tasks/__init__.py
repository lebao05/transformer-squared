from .arc import AI2ArcTask
from .base import FewShotTask, Task
from .hellaswag import HellaSwagTask
from .math import MathTask
from .mmlu import MMLUTask

try:
    from .gsm8k import Gsm8kTask
except ImportError:
    pass

try:
    from .mbpp2 import Mbpp2Task
except ImportError:
    pass

try:
    from .cls import ClsTask
except ImportError:
    pass
