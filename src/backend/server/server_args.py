import argparse

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Lattica configuration
    parser.add_argument("--initial-peers", nargs="+", default=[], help="List of initial DHT peers")
    parser.add_argument("--relay-servers", nargs="+", default=[], help="List of relay DHT peers")
    parser.add_argument(
        "--announce-maddrs", nargs="+", default=[], help="List of multiaddresses to announce"
    )
    parser.add_argument("--tcp-port", type=int, default=0, help="Port for Lattica TCP listening")
    parser.add_argument("--udp-port", type=int, default=0, help="Port for Lattica UDP listening")
    parser.add_argument("--dht-prefix", type=str, default="gradient", help="Prefix for DHT keys")

    # Scheduler configuration
    parser.add_argument("--host", type=str, default="localhost", help="Host to listen on")
    parser.add_argument("--port", type=int, default=3001, help="Port to listen on")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Model name")
    parser.add_argument("--init-nodes-num", type=int, default=None, help="Number of initial nodes")
    parser.add_argument(
        "--is-local-network", type=bool, default=True, help="Whether to use local network"
    )
    parser.add_argument(
        "--use-hfcache",
        action="store_true",
        default=False,
        help="Use local Hugging Face cache only (no network download)",
    )

    args = parser.parse_args()

    return args
