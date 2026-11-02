#!/usr/bin/env python3
"""Query and display Kiro models with their credit usage rates."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

logger.disable("kiro")

from kiro.auth import KiroAuthManager
from kiro.utils import get_kiro_headers
from query_credits import DEFAULT_DB_PATH, read_api_region

MODELS_PATH = "/ListAvailableModels"
MODEL_COLUMN_WIDTH = 24
CREDIT_COLUMN_WIDTH = 22


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Query Kiro models and their credit usage rates."
    )
    parser.add_argument(
        "--db",
        default=os.getenv("KIRO_CLI_DB_FILE", DEFAULT_DB_PATH),
        help=f"kiro-cli SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--proxy",
        default=os.getenv("VPN_PROXY_URL"),
        help="HTTP or SOCKS5 proxy URL, e.g. http://127.0.0.1:7892",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete API response as JSON",
    )
    return parser.parse_args()


async def query_models(db_path: str, proxy: Optional[str]) -> Dict[str, Any]:
    """Fetch all available Kiro models and their metadata.

    Args:
        db_path: Path to the kiro-cli SQLite database.
        proxy: Optional HTTP or SOCKS5 proxy URL.

    Returns:
        Complete JSON response from ``/ListAvailableModels``.

    Raises:
        FileNotFoundError: If the SQLite database does not exist.
        httpx.HTTPError: If the model request fails.
        ValueError: If authentication cannot provide a token.
    """
    api_region = read_api_region(Path(db_path).expanduser())
    auth_manager = KiroAuthManager(
        sqlite_db=db_path,
        api_region=api_region,
    )
    params: Dict[str, str] = {"origin": "AI_EDITOR"}
    if auth_manager.profile_arn:
        params["profileArn"] = auth_manager.profile_arn

    client_options: Dict[str, Any] = {"timeout": 30.0}
    if proxy:
        client_options["proxy"] = proxy

    async with httpx.AsyncClient(**client_options) as client:
        next_token: Optional[str] = None
        models: List[Dict[str, Any]] = []
        default_model: Optional[Dict[str, Any]] = None

        while True:
            request_params = dict(params)
            if next_token:
                request_params["nextToken"] = next_token

            access_token = await auth_manager.get_access_token()
            response = await client.get(
                f"{auth_manager.q_host}{MODELS_PATH}",
                params=request_params,
                headers=get_kiro_headers(auth_manager, access_token),
            )
            if response.status_code == 403:
                access_token = await auth_manager.force_refresh()
                response = await client.get(
                    f"{auth_manager.q_host}{MODELS_PATH}",
                    params=request_params,
                    headers=get_kiro_headers(auth_manager, access_token),
                )
            response.raise_for_status()
            data = response.json()
            models.extend(data.get("models", []))
            default_model = data.get("defaultModel") or default_model
            next_token = data.get("nextToken")
            if not next_token:
                return {
                    "models": models,
                    "defaultModel": default_model,
                }


def format_rate(model: Dict[str, Any]) -> str:
    """Format a model's credit usage rate.

    Args:
        model: Model metadata returned by Kiro.

    Returns:
        Rate formatted like ``0.60x credits``.
    """
    multiplier = model.get("rateMultiplier")
    unit = model.get("rateUnit") or "credits"
    if multiplier is None:
        return "unknown"
    normalized_unit = "credits" if unit.lower() in {"credit", "credits"} else unit
    return f"{float(multiplier):.2f}x {normalized_unit}"


def print_models(response: Dict[str, Any]) -> None:
    """Print models in the same three-column style as ``kiro-cli /model``.

    Args:
        response: JSON response returned by ``/ListAvailableModels``.
    """
    models = response.get("models", [])
    default_model = (response.get("defaultModel") or {}).get("modelId")

    print(
        f"{'Model':<{MODEL_COLUMN_WIDTH}}"
        f"{'Credit usage':<{CREDIT_COLUMN_WIDTH}}"
        "Description"
    )
    for model in models:
        model_id = model.get("modelId", "unknown")
        marker = "> " if model_id == default_model else "  "
        model_name = f"{marker}{model_id}"
        print(
            f"{model_name:<{MODEL_COLUMN_WIDTH}}"
            f"{format_rate(model):<{CREDIT_COLUMN_WIDTH}}"
            f"{model.get('description') or ''}"
        )


def main() -> int:
    """Run the model rate query.

    Returns:
        Process exit code.
    """
    args = parse_args()
    try:
        response = asyncio.run(query_models(args.db, args.proxy))
        if args.json:
            print(json.dumps(response, ensure_ascii=False, indent=2))
        else:
            print_models(response)
        return 0
    except (FileNotFoundError, ValueError, httpx.HTTPError, OSError) as error:
        print(f"查询失败: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
