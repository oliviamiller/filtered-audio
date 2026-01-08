import asyncio
from viam.module.module import Module
try:
    from models.wake_word_filter import WakeWordFilter
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.wake_word_filter import WakeWordFilter


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
