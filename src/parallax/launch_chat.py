from parallax.server.node_chat_http_server import run_node_chat_http_server
from parallax.server.server_args import parse_args
from parallax_utils.logging_config import get_logger, set_log_level

logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        args = parse_args()
        set_log_level(args.log_level)
        logger.debug(f"args: {args}")

        run_node_chat_http_server(args)
    except KeyboardInterrupt:
        logger.debug("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(e)
