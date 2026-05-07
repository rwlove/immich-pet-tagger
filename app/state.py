import asyncio

scan_lock: asyncio.Lock | None = None


def init():
    global scan_lock
    scan_lock = asyncio.Lock()
