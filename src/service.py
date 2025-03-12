import logging
import asyncio
async def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting service")

if __name__ == "__main__":
    asyncio.run(main())
