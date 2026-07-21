import asyncio

from fastapi import APIRouter

from runner.nodes.tts.piper_catalog import fetch_piper_catalog


router = APIRouter(prefix="/piper", tags=["piper"])


@router.get("/voices")
async def list_piper_voices() -> dict[str, dict]:
    voices = await asyncio.to_thread(fetch_piper_catalog)
    return {voice.key: voice.model_dump(mode="json") for voice in voices}
