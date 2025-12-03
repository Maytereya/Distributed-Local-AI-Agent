import asyncio


async def main():
    from agent_logic_2.nayka_api.api_nayka import (
        get_all_doctors,
        save_doctors_data,
        get_active_date_str,
    )

    docs = await asyncio.to_thread(get_all_doctors)
    save_doctors_data(docs)
    print(f"✅ Refreshed doctors cache for active date {get_active_date_str()}; total: {len(docs)}")


if __name__ == "__main__":
    asyncio.run(main())

