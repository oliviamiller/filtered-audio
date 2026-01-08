import asyncio
from viam.module.module import Module
try:
    from models.trigger import Trigger
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.trigger import Trigger


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
