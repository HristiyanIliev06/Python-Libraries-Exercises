import asyncio
import random

async def fetch_item(item_id: int) -> dict:
    await asyncio.sleep(random.uniform(0.5, 2.00))

    if random.random() < 0.1:
        raise Exception(f"Failed to fetch item {item_id}")

    return {"id": item_id, "data": f"payload-{item_id}"}

async def fetch_all(item_ids: list[int], failed_to_fetch: list[str]) -> list[dict]:
    successful_to_fetch: list[dict] = []
    
    for item_id in item_ids:
        try:
            item = await fetch_item(item_id)
            successful_to_fetch.append(item)
        except Exception as e:
            failed_to_fetch.append(e)

    return successful_to_fetch

"""Demonstrate with 15 item IDs. Print how many succeeded and how many failed."""

item_ids = list(range(1, 16))
failed_to_fetch: list[str] = []

async def main():
    # successful_to_fetch = await fetch_all(item_ids, failed_to_fetch) # All items are distributed depending on if they succeed or fail to fetch
    successful_to_fetch = await asyncio.gather(fetch_all(item_ids, failed_to_fetch), return_exceptions=True) # A portion of all items do not fall in either category and that's it
    # One case: 1 succeeds & 2 fail of 15; Another case: 1 succeeds & 5 fail of 15
    print(f"Number of items fetched successfully: {len(successful_to_fetch)}")
    print(f"Number of items failed to fetch: {len(failed_to_fetch)}")

asyncio.run(main())
