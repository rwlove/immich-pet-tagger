import asyncio

scan_lock: asyncio.Lock | None = None
neg_progress: dict = {"current": 0, "total": 0, "running": False}


def init():
    global scan_lock
    scan_lock = asyncio.Lock()
