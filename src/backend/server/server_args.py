import argparse

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # P2P configuration
    parser.add_argument("--initial-peers", nargs="+", default=[], help="List of initial DHT peers")

    parser.add_argument("--relay-servers", nargs="+", default=[], help="List of relay DHT peers")

    parser.add_argument(
        "--announce-maddrs", nargs="+", default=[], help="List of multiaddresses to announce"
    )

    parser.add_argument("--dht-port", type=int, default=None, help="Port for DHT communication")

    parser.add_argument("--host-maddrs", type=str, default=None, help="Multiaddress to host")

    parser.add_argument("--dht-prefix", type=str, default="gradient", help="Prefix for DHT keys")

    parser.add_argument("--public-ip", type=str, default=None, help="Public IP address to announce")

    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")

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

    args = parser.parse_args()

    return args
