import argparse
import random
import time
from typing import Any, Dict, Optional

import requests


def simulate_sensor_data(include_vibration: bool = True) -> Dict[str, Any]:
    temperature = round(random.uniform(35.0, 55.0), 1)
    gas_level = random.choice(["low", "moderate", "high methane"])

    sensor_data: Dict[str, Any] = {
        "temperature": temperature,
        "gas_level": gas_level,
    }

    if include_vibration:
        sensor_data["vibration"] = random.choice(["normal", "elevated", "severe"])

    return sensor_data


def build_prompt(sensor_data: Dict[str, Any]) -> str:
    temperature = sensor_data.get("temperature", "unknown")
    gas_level = sensor_data.get("gas_level", "unknown")
    vibration = sensor_data.get("vibration")

    prompt = (
        f"Given the following mining sensor data: temperature={temperature}°C, "
        f"gas={gas_level}"
    )

    if vibration is not None:
        prompt += f", vibration={vibration}"

    prompt += ", what risk does this indicate?"
    return prompt


def call_api(
    server_url: str,
    payload: Dict[str, Any],
    result_base_url: str,
    timeout_seconds: float = 10.0,
    max_attempts: int = 3,
    poll_interval: float = 2.0,
    poll_max_attempts: int = 150,
) -> Dict[str, Any]:
    total_start = time.perf_counter()

    # Phase 1 — submit query, get task_id
    task_id: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(server_url, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "ok" and "task_id" in data:
                task_id = data["task_id"]
                break
            raise ValueError(f"unexpected response: {data}")
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as e:
            if attempt < max_attempts:
                backoff = 2 ** (attempt - 1)
                print(f"Submit attempt {attempt} failed: {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                return {"ok": False, "error": str(e), "round_trip_time": time.perf_counter() - total_start}

    if task_id is None:
        return {"ok": False, "error": "failed to submit query", "round_trip_time": time.perf_counter() - total_start}

    print(f"Task submitted (task_id={task_id})")

    # Phase 2 — poll for result
    result_url = f"{result_base_url}/{task_id}"
    for attempt in range(1, poll_max_attempts + 1):
        try:
            response = requests.get(result_url, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            status = data.get("status")
            elapsed = time.perf_counter() - total_start
            if status == "completed":
                return {
                    "ok": True,
                    "model_response": data.get("response", ""),
                    "round_trip_time": elapsed,
                }
            elif status == "failed":
                return {
                    "ok": False,
                    "error": data.get("error", "inference failed"),
                    "round_trip_time": elapsed,
                }
            print(f"  status={status}, elapsed={elapsed:.1f}s")
            time.sleep(poll_interval)
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            if attempt < poll_max_attempts:
                backoff = min(2 ** (attempt - 1), 10)
                print(f"Poll attempt {attempt} failed: {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                return {"ok": False, "error": str(e), "round_trip_time": time.perf_counter() - total_start}

    return {"ok": False, "error": "timed out waiting for result", "round_trip_time": time.perf_counter() - total_start}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jetson client for sending mining sensor prompts to a FastAPI server."
    )
    parser.add_argument(
        "--agx-ip",
        required=True,
        help="IP address of the AGX Orin running the FastAPI server.",
    )
    parser.add_argument(
        "--endpoint",
        default="/query",
        help="API endpoint path on the server. Default: /query",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds. Default: 10",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Maximum retry attempts. Default: 3",
    )
    parser.add_argument(
        "--no-vibration",
        action="store_true",
        help="Disable vibration in the simulated sensor payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensor_data = simulate_sensor_data(include_vibration=not args.no_vibration)
    prompt = build_prompt(sensor_data)
    base_url = f"http://{args.agx_ip}:8000"
    server_url = f"{base_url}{args.endpoint}"
    result_base_url = f"{base_url}/result"
    payload = {
        "prompt": prompt,
        "sensor": sensor_data,
    }

    print("=== Sensor Client ===")
    print(f"Server: {server_url}")
    print(f"Prompt: {prompt}")
    print(f"Sensor data: {sensor_data}")
    print()

    result = call_api(
        server_url=server_url,
        payload=payload,
        result_base_url=result_base_url,
        timeout_seconds=args.timeout,
        max_attempts=args.attempts,
    )

    print("=== Result ===")
    if result["ok"]:
        print(f"Status: success")
        print(f"Model response: {result['model_response']}")
        print(f"Client round-trip time: {result['round_trip_time']:.3f}s")
    else:
        print(f"Status: failed")
        print(f"Error: {result['error']}")
        if result.get("round_trip_time") is not None:
            print(f"Client round-trip time: {result['round_trip_time']:.3f}s")


if __name__ == "__main__":
    main()