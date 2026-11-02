#!/usr/bin/env python3
"""Query the Kiro account credit usage and remaining balance."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from loguru import logger

logger.disable("kiro")

from kiro.auth import KiroAuthManager

DEFAULT_DB_PATH = "~/.local/share/kiro-cli/data.sqlite3"
DEFAULT_API_REGION = "us-east-1"
USAGE_PATH = "/getUsageLimits"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Query Kiro credits usage and remaining balance."
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


def read_api_region(db_path: Path) -> str:
    """Read the API region from the kiro-cli profile state.

    Args:
        db_path: Path to the kiro-cli SQLite database.

    Returns:
        API region from the profile ARN, or the default region.
    """
    try:
        with sqlite3.connect(str(db_path)) as connection:
            row = connection.execute(
                "SELECT value FROM state WHERE key = ?",
                ("api.codewhisperer.profile",),
            ).fetchone()
        if row:
            profile = json.loads(row[0])
            arn = profile.get("arn", "")
            parts = arn.split(":")
            if len(parts) >= 4 and parts[3]:
                return parts[3]
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError):
        pass
    return DEFAULT_API_REGION


def get_usage_record(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the credit usage record from the Kiro response.

    Args:
        response: JSON response returned by ``/getUsageLimits``.

    Returns:
        Credit usage record.

    Raises:
        ValueError: If the response does not contain a credit usage record.
    """
    records = response.get("usageBreakdownList") or []
    for record in records:
        if record.get("resourceType") == "CREDIT":
            return record
    raise ValueError("The Kiro response does not contain a CREDIT usage record.")


def get_number(record: Dict[str, Any], precise_key: str, fallback_key: str) -> float:
    """Read a precise numeric value with an integer fallback.

    Args:
        record: Credit usage record.
        precise_key: Preferred precise field name.
        fallback_key: Fallback field name.

    Returns:
        Numeric value.
    """
    value = record.get(precise_key)
    if value is None:
        value = record.get(fallback_key)
    if value is None:
        raise ValueError(f"Missing usage field: {precise_key}")
    return float(value)


def format_reset_time(timestamp: Optional[float]) -> str:
    """Format the next reset timestamp in the local timezone.

    Args:
        timestamp: Unix timestamp, or None.

    Returns:
        Human-readable reset time, or ``unknown``.
    """
    if timestamp is None:
        return "unknown"
    reset_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone()
    return reset_time.strftime("%Y-%m-%d %H:%M:%S %Z")


def format_number(value: float) -> str:
    """Format a credit value without unnecessary trailing zeroes.

    Args:
        value: Credit value.

    Returns:
        Formatted number.
    """
    return f"{value:.2f}".rstrip("0").rstrip(".")


async def query_usage(
    db_path: Path,
    proxy: Optional[str],
) -> Dict[str, Any]:
    """Refresh authentication if needed and query Kiro credit usage.

    Args:
        db_path: Path to the kiro-cli SQLite database.
        proxy: Optional HTTP or SOCKS5 proxy URL.

    Returns:
        Complete JSON response from Kiro.

    Raises:
        FileNotFoundError: If the SQLite database does not exist.
        httpx.HTTPError: If the usage request fails.
        ValueError: If credentials or the response are invalid.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"kiro-cli database not found: {db_path}")

    api_region = read_api_region(db_path)
    auth_manager = KiroAuthManager(
        sqlite_db=str(db_path),
        api_region=api_region,
    )
    access_token = await auth_manager.get_access_token()
    usage_url = f"https://q.{api_region}.amazonaws.com{USAGE_PATH}"
    headers = {"Authorization": f"Bearer {access_token}"}

    client_options: Dict[str, Any] = {"timeout": 30.0}
    if proxy:
        client_options["proxy"] = proxy
    async with httpx.AsyncClient(**client_options) as client:
        response = await client.get(usage_url, headers=headers)
        response.raise_for_status()
        return response.json()


def print_usage(response: Dict[str, Any]) -> None:
    """Print a concise credit usage summary.

    Args:
        response: JSON response returned by ``/getUsageLimits``.
    """
    record = get_usage_record(response)
    current = get_number(record, "currentUsageWithPrecision", "currentUsage")
    limit = get_number(record, "usageLimitWithPrecision", "usageLimit")
    remaining = limit - current
    percentage = current / limit * 100 if limit else 0

    print(f"订阅: {response.get('subscriptionInfo', {}).get('subscriptionTitle', 'unknown')}")
    print(f"已使用 credits: {format_number(current)}")
    print(f"总额度 credits: {format_number(limit)}")
    print(f"剩余 credits: {format_number(remaining)}")
    print(f"使用率: {percentage:.2f}%")
    print(f"下次重置: {format_reset_time(record.get('nextDateReset'))}")
    print(f"超额状态: {response.get('overageConfiguration', {}).get('overageStatus', 'unknown')}")


def main() -> int:
    """Run the credit usage query.

    Returns:
        Process exit code.
    """
    args = parse_args()
    db_path = Path(args.db).expanduser()

    try:
        response = asyncio.run(query_usage(db_path, args.proxy))
        if args.json:
            print(json.dumps(response, ensure_ascii=False, indent=2))
        else:
            print_usage(response)
        return 0
    except (FileNotFoundError, ValueError, httpx.HTTPError, OSError) as error:
        print(f"查询失败: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
