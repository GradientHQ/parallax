"""
Launch the Parallax server.

This script is used to launch the Parallax server.
It will start the P2P server and the executor.

Example command:
python src/parallax/launch.py \
    --model-path Qwen/Qwen3-0.6B-MLX-bf16 \
    --max-num-tokens-per-batch 16384 \
    --max-batch-size 128 \
    --start-layer 14 \
    --end-layer 28 \
    --initial-peers {peer of GPU which hold the first half model}
"""

import multiprocessing
import tempfile

from parallax.p2p.server import ServerState, launch_p2p_server
from parallax.server.executor import Executor
from parallax.server.http_server import launch_http_server
from parallax.server.server_args import parse_args
from parallax_utils.logging_config import get_logger

logger = get_logger("parallax.launch")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    try:
        args = parse_args()
        logger.debug(f"args: {args}")

        args.recv_from_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.send_to_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.executor_input_ipc = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.executor_output_ipc = f"ipc://{tempfile.NamedTemporaryFile().name}"

        logger.debug(f"executor_input_addr: {args.executor_input_ipc}")
        logger.debug(f"executor_output_addr: {args.executor_output_ipc}")
        gradient_server = None
        if args.scheduler_addr is None:
            # only launch http server on head node
            if args.start_layer == 0:
                launch_http_server(args)
            executor = Executor.create_from_args(args)
            launch_p2p_server(
                initial_peers=args.initial_peers,
                scheduler_addr=args.scheduler_addr,
                relay_servers=args.relay_servers,
                pp_start_layer=args.start_layer,
                pp_end_layer=args.end_layer,
                hidden_layers=executor.config.get("num_hidden_layers"),
                dht_port=args.dht_port,
                dht_prefix=args.dht_prefix,
                host_maddrs=args.host_maddrs,
                announce_maddrs=args.announce_maddrs,
                http_port=args.port if args.announce_http_port is None else args.announce_http_port,
                notify_url=args.notify_url,
                recv_from_peer_addr=args.recv_from_peer_addr,
                send_to_peer_addr=args.send_to_peer_addr,
                model_name=args.model_path,
                max_batch_size=args.max_batch_size,
                max_sequence_length=args.max_sequence_length,
            )
        else:
            gradient_server = launch_p2p_server(
                initial_peers=args.initial_peers,
                scheduler_addr=args.scheduler_addr,
                relay_servers=args.relay_servers,
                pp_start_layer=None,
                pp_end_layer=None,
                hidden_layers=None,
                dht_port=args.dht_port,
                dht_prefix=args.dht_prefix,
                host_maddrs=args.host_maddrs,
                announce_maddrs=args.announce_maddrs,
                http_port=args.port if args.announce_http_port is None else args.announce_http_port,
                notify_url=args.notify_url,
                recv_from_peer_addr=args.recv_from_peer_addr,
                send_to_peer_addr=args.send_to_peer_addr,
                model_name=args.model_path,
                max_batch_size=args.max_batch_size,
                max_sequence_length=args.max_sequence_length,
            )
            args.start_layer = gradient_server.block_start_index
            args.end_layer = gradient_server.block_end_index
            args.model_path = gradient_server.model_name
            logger.debug(
                f"Start Executor with start_layer: {args.start_layer}, end_layer: {args.end_layer}"
            )
            gradient_server.status = ServerState.INITIALIZING
            # only launch http server on head node
            if args.start_layer == 0:
                launch_http_server(args)
            executor = Executor.create_from_args(args)

        try:
            if gradient_server is not None:
                gradient_server.status = ServerState.READY
            executor.run_loop()
        except KeyboardInterrupt:
            logger.debug("Received interrupt signal, shutting down...")
        finally:
            if gradient_server is not None:
                gradient_server.shutdown()
            executor.shutdown()

    except Exception as e:
        logger.exception(e)
