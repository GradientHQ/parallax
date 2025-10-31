"""
Launch the Parallax server.

This script is used to launch the Parallax server.
It will start the P2P server and the executor.

Example command:
python src/parallax/launch.py \
    --model-path Qwen/Qwen3-0.6B \
    --max-num-tokens-per-batch 16384 \
    --max-batch-size 128 \
    --start-layer 0 \
    --end-layer 28
"""

import multiprocessing
import os
import tempfile
import threading

from common.version_check import check_latest_release
from parallax.p2p.server import ServerState, launch_p2p_server
from parallax.server.executor import Executor
from parallax.server.http_server import launch_http_server
from parallax.server.server_args import parse_args
from parallax_utils.ascii_anime import display_parallax_join
from parallax_utils.logging_config import get_logger, set_log_level

logger = get_logger("parallax.launch")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    gradient_server = None
    http_server_process = None
    executor = None
    try:
        args = parse_args()
        set_log_level(args.log_level)
        logger.debug(f"args: {args}")
        args.recv_from_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.send_to_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.executor_input_ipc = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.executor_output_ipc = f"ipc://{tempfile.NamedTemporaryFile().name}"

        # Silence tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        logger.debug(f"executor_input_addr: {args.executor_input_ipc}")
        logger.debug(f"executor_output_addr: {args.executor_output_ipc}")
        if args.scheduler_addr is None:
            if args.log_level != "DEBUG":
                display_parallax_join(args.model_path)
            check_latest_release()

            # only launch http server on head node
            if args.start_layer == 0:
                http_server_process = launch_http_server(args)
            executor = Executor.create_from_args(args)
            launch_p2p_server(
                initial_peers=args.initial_peers,
                scheduler_addr=args.scheduler_addr,
                relay_servers=args.relay_servers,
                pp_start_layer=args.start_layer,
                pp_end_layer=args.end_layer,
                hidden_layers=executor.config.get("num_hidden_layers"),
                tcp_port=args.tcp_port,
                udp_port=args.udp_port,
                dht_prefix=args.dht_prefix,
                announce_maddrs=args.announce_maddrs,
                http_port=args.port,
                notify_url=args.notify_url,
                recv_from_peer_addr=args.recv_from_peer_addr,
                send_to_peer_addr=args.send_to_peer_addr,
                model_name=args.model_path,
                max_batch_size=args.max_batch_size,
                max_sequence_length=args.max_sequence_length,
                param_hosting_ratio=args.param_hosting_ratio,
                kv_cache_ratio=args.kv_cache_ratio,
            )
        else:
            gradient_server = launch_p2p_server(
                initial_peers=args.initial_peers,
                scheduler_addr=args.scheduler_addr,
                relay_servers=args.relay_servers,
                pp_start_layer=None,
                pp_end_layer=None,
                hidden_layers=None,
                tcp_port=args.tcp_port,
                udp_port=args.udp_port,
                dht_prefix=args.dht_prefix,
                announce_maddrs=args.announce_maddrs,
                http_port=args.port,
                notify_url=args.notify_url,
                recv_from_peer_addr=args.recv_from_peer_addr,
                send_to_peer_addr=args.send_to_peer_addr,
                model_name=args.model_path,
                max_batch_size=args.max_batch_size,
                max_sequence_length=args.max_sequence_length,
                param_hosting_ratio=args.param_hosting_ratio,
                kv_cache_ratio=args.kv_cache_ratio,
            )
            args.start_layer = gradient_server.block_start_index
            args.end_layer = gradient_server.block_end_index
            args.model_path = gradient_server.model_name
            # Hard code for mlx-community models
            logger.debug(
                f"Start Executor with start_layer: {args.start_layer}, end_layer: {args.end_layer}"
            )
            gradient_server.status = ServerState.INITIALIZING

            if args.log_level != "DEBUG":
                display_parallax_join(args.model_path)
            check_latest_release()

            # only launch http server on head node
            if args.start_layer == 0:
                http_server_process = launch_http_server(args)
            executor = Executor.create_from_args(args)

        if gradient_server is not None:
            gradient_server.status = ServerState.READY
        executor.run_loop()
    except KeyboardInterrupt:
        logger.debug("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(e)
    finally:
        t = None
        if http_server_process is not None:

            def terminate_http_server_process(process):
                logger.debug("Terminating HTTP server process...")
                try:
                    process.kill()
                    process.join()
                except Exception as e:
                    logger.error(f"Failed to terminate HTTP server process: {e}")

            if http_server_process is not None:
                t = threading.Thread(
                    target=terminate_http_server_process, args=(http_server_process,)
                )
                t.start()
        if gradient_server is not None:
            gradient_server.shutdown()
        if executor is not None:
            executor.shutdown()
        if t is not None:
            t.join()
