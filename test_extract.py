import asyncio
from app.services.youtube_service import extract_youtube

async def main():
    try:
        url = "https://www.youtube.com/watch?v=Q0BOH_s9gSU"
        data = await extract_youtube(url)
        print("Success:", data.title)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
