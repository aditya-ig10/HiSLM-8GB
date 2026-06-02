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


def _extract_response_fields(response_json: Any) -> tuple[Optional[str], Optional[Any]]:
    if isinstance(response_json, dict):
        model_response = (
            response_json.get("response")
            or response_json.get("model_response")
            or response_json.get("answer")
            or response_json.get("result")
        )
        server_latency = (
            response_json.get("latency")
            or response_json.get("server_latency")
            or response_json.get("processing_time")
        )
        return model_response, server_latency

    return None, None


def call_api(
    server_url: str,
    payload: Dict[str, Any],
    timeout_seconds: float = 10.0,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    last_error: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        start_time = time.perf_counter()
        try:
            response = requests.post(server_url, json=payload, timeout=timeout_seconds)
            round_trip_time = time.perf_counter() - start_time
            response.raise_for_status()

            try:
                response_json = response.json()
            except ValueError:
                response_json = {"raw_response": response.text}

            model_response, server_latency = _extract_response_fields(response_json)
            if model_response is None:
                model_response = response_json.get("raw_response", response.text)

            return {
                "ok": True,
                "attempt": attempt,
                "model_response": model_response,
                "server_latency": server_latency,
                "round_trip_time": round_trip_time,
                "status_code": response.status_code,
            }
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as error:
            last_error = error
            if attempt < max_attempts:
                backoff_seconds = 2 ** (attempt - 1)
                print(f"Attempt {attempt} failed: {error}. Retrying in {backoff_seconds}s...")
                time.sleep(backoff_seconds)
            else:
                round_trip_time = time.perf_counter() - start_time
                return {
                    "ok": False,
                    "attempt": attempt,
                    "error": str(error),
                    "round_trip_time": round_trip_time,
                }

    return {
        "ok": False,
        "attempt": max_attempts,
        "error": str(last_error) if last_error else "Unknown error",
        "round_trip_time": None,
    }


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
    server_url = f"http://{args.agx_ip}:8000{args.endpoint}"
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
        timeout_seconds=args.timeout,
        max_attempts=args.attempts,
    )

    print("=== Result ===")
    if result["ok"]:
        print(f"Status: success (attempt {result['attempt']})")
        print(f"Model response: {result['model_response']}")
        if result["server_latency"] is not None:
            print(f"Server latency: {result['server_latency']}")
        else:
            print("Server latency: not provided by server")
        print(f"Client round-trip time: {result['round_trip_time']:.3f}s")
    else:
        print(f"Status: failed after {result['attempt']} attempt(s)")
        print(f"Error: {result['error']}")
        if result["round_trip_time"] is not None:
            print(f"Client round-trip time: {result['round_trip_time']:.3f}s")


if __name__ == "__main__":
    main()