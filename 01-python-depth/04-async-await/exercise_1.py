
import asyncio


async def task(name: str, delay: float) -> str:
    print(f"{name} starting")
    await asyncio.sleep(delay)
    print(f"{name} done")
    return f"{name} finished after {delay}s"

async def main():
    results = await asyncio.gather(
        task("A", 2),
        task("B", 1),
        task("C", 3),
    )
    print(results)

asyncio.run(main())