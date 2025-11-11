"""
Launch the Parallax server.

This script is used to launch the Parallax server.
It will start the following services:
    1.Executor with tp_rank=0 in the main process.
    2.Executor with tp_rank>0, each tp_rank as a subprocess.
    3.HTTP server as a subprocess.
    4.P2P server as a thread in the main process.

Example command:
python src/parallax/launch.py \
    --model-path Qwen/Qwen3-0.6B \
    --max-num-tokens-per-batch 16384 \
    --max-batch-size 128 \
    --start-layer 0 \
    --end-layer 28
"""

import argparse
import multiprocessing
import os
import tempfile
import threading

from parallax.p2p.server import ServerState, launch_p2p_server
from parallax.server.executor import (
    Executor,
    run_executor_process,
    stop_executor_process,
)
from parallax.server.http_server import launch_http_server, stop_http_server
from parallax.server.server_args import parse_args
from parallax.utils.utils import fetch_model_from_hf, initialize_nccl_port
from parallax_utils.ascii_anime import display_parallax_join
from parallax_utils.logging_config import get_logger, set_log_level
from parallax_utils.version_check import check_latest_release

logger = get_logger("parallax.launch")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    gradient_server = None
    http_server_process = None
    executor = None
    executor_subprocs = []
    try:
        args = parse_args()
        set_log_level(args.log_level)
        logger.debug(f"args: {args}")
        args.recv_from_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.send_to_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.executor_input_ipc = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.executor_output_ipc = f"ipc://{tempfile.NamedTemporaryFile().name}"
        if args.nccl_port is None:
            args.nccl_port = initialize_nccl_port()

        # Silence tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        logger.debug(f"executor_input_addr: {args.executor_input_ipc}")
        logger.debug(f"executor_output_addr: {args.executor_output_ipc}")
        logger.debug(f"nccl_port: {args.nccl_port}")
        if args.scheduler_addr is None:
            if args.log_level != "DEBUG":
                display_parallax_join(args.model_path)
            check_latest_release()

            config = fetch_model_from_hf(args.model_path)
            # only launch http server on head node
            if args.start_layer == 0:
                http_server_process = launch_http_server(args)
            launch_p2p_server(
                initial_peers=args.initial_peers,
                scheduler_addr=args.scheduler_addr,
                relay_servers=args.relay_servers,
                pp_start_layer=args.start_layer,
                pp_end_layer=args.end_layer,
                hidden_layers=config.get("num_hidden_layers"),
                tp_size=args.tp_size,
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
            if gradient_server is not None:
                gradient_server.status = ServerState.READY

            # For each tp_rank > 0, create a subprocess and run executor
            tp_rank_range = range(args.tp_size)
            for tp_rank in (1, tp_rank_range):
                args_copy = argparse.Namespace(**vars(args))
                args_copy.tp_rank = tp_rank
                proc = multiprocessing.Process(
                    target=run_executor_process,
                    args=(args_copy,),
                )
                proc.start()
                executor_subprocs.append(proc)
            # Launch executor with tp_rank=0 in the main process
            args.tp_rank = 0
            executor = Executor.create_from_args(args)
            executor.run_loop()
        else:
            gradient_server = launch_p2p_server(
                initial_peers=args.initial_peers,
                scheduler_addr=args.scheduler_addr,
                relay_servers=args.relay_servers,
                pp_start_layer=args.start_layer,
                pp_end_layer=args.end_layer,
                hidden_layers=None,
                tp_size=args.tp_size,
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
            args.tp_size = gradient_server.tp_size

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

            # Main execution loop with layer reallocation support
            while True:
                try:
                    # For each tp_rank > 0, create a subprocess and run executor
                    tp_rank_range = range(args.tp_size)
                    for tp_rank in (1, tp_rank_range):
                        args_copy = argparse.Namespace(**vars(args))
                        args_copy.tp_rank = tp_rank
                        proc = multiprocessing.Process(
                            target=run_executor_process,
                            args=(args_copy,),
                        )
                        proc.start()
                        executor_subprocs.append(proc)
                    # Launch executor with tp_rank=0 in the main process
                    args.tp_rank = 0
                    executor = Executor.create_from_args(args, gradient_server=gradient_server)
                    if gradient_server is not None:
                        gradient_server.status = ServerState.READY

                    executor.run_loop()

                    # Check if layer allocation changed (executor exited due to reallocation)
                    if gradient_server is not None and gradient_server._layer_allocation_changed:
                        logger.warning(
                            "Layer allocation changed! Reloading executor with new layers..."
                        )

                        # shutdown all executor processes
                        thread_pool = []
                        for executor_process in executor_subprocs:
                            t = threading.Thread(
                                target=stop_executor_process, args=(executor_process,)
                            )
                            t.start()
                            thread_pool.append(t)
                        executor.shutdown()
                        for t in thread_pool:
                            t.join()

                        if args.start_layer == 0:
                            http_server_process = stop_http_server(http_server_process)
                        if gradient_server.block_start_index == 0:
                            http_server_process = launch_http_server(args)

                        # Update args with new layer allocation
                        args.start_layer = gradient_server.block_start_index
                        args.end_layer = gradient_server.block_end_index
                        if gradient_server.model_name:
                            args.model_path = gradient_server.model_name

                        logger.info(
                            f"Creating new executor with layers [{args.start_layer}, {args.end_layer})"
                        )

                        gradient_server._layer_allocation_changed = False
                        continue  # Create new executor in next iteration
                    else:
                        break  # Normal exit
                except KeyboardInterrupt:
                    logger.debug("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    logger.exception(f"Executor error: {e}")
                    # If layer allocation changed, try to reload
                    if gradient_server is not None and gradient_server._layer_allocation_changed:
                        logger.info("Attempting to reload executor after error...")
                        if executor is not None:
                            executor.shutdown()
                        continue
                    else:
                        raise
    except KeyboardInterrupt:
        logger.debug("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(e)
    finally:
        thread_pool = []

        # Shutdown http server
        if http_server_process is not None:
            t = threading.Thread(target=stop_http_server, args=(http_server_process,))
            t.start()
            thread_pool.append(t)

        # Shutdown gradient server
        if gradient_server is not None:
            gradient_server.shutdown()

        # Shutdown executor subprocesses
        for executor_process in executor_subprocs:
            t = threading.Thread(target=stop_executor_process, args=(executor_process,))
            t.start()
            thread_pool.append(t)

        # Shutdown executor main process
        if executor is not None:
            executor.shutdown()

        for t in thread_pool:
            t.join()
