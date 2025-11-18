"""
Launch the Parallax server.

This script is used to launch the Parallax server.
It will start the following services:
    1.Executor with tp_rank=0 as a subprocess.
    2.Executor with tp_rank>0, each tp_rank as a subprocess.
    3.HTTP server as a subprocess.
    4.P2P server as a subprocess.

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
import time

from parallax.p2p.server import ServerState, launch_p2p_server_process, stop_p2p_server
from parallax.server.executor import run_executor_process, stop_executor_process
from parallax.server.http_server import launch_http_server, stop_http_server
from parallax.server.server_args import parse_args
from parallax.utils.utils import fetch_model_from_hf, initialize_nccl_port
from parallax_utils.ascii_anime import display_parallax_join
from parallax_utils.logging_config import get_logger, set_log_level
from parallax_utils.version_check import check_latest_release

logger = get_logger("parallax.launch")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    p2p_server_process = None
    http_server_process = None
    executor_subprocs = []
    # Shared state for layer allocation info (used when P2P server is in subprocess)
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    shared_state["block_start_index"] = None
    shared_state["block_end_index"] = None
    shared_state["model_name"] = None
    shared_state["tp_size"] = None
    shared_state["_layer_allocation_changed"] = False
    shared_state["status"] = ServerState.JOINING.value

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
            # Launch P2P server as subprocess
            p2p_server_process = launch_p2p_server_process(
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
                param_mem_ratio=args.param_mem_ratio,
                kvcache_mem_ratio=args.kvcache_mem_ratio,
                shared_state=shared_state,
            )

            # Launch all executor processes (including tp_rank=0)
            for tp_rank in range(args.tp_size):
                args_copy = argparse.Namespace(**vars(args))
                args_copy.tp_rank = tp_rank
                proc = multiprocessing.Process(
                    target=run_executor_process,
                    args=(
                        args_copy,
                        None,
                        shared_state,
                    ),  # gradient_server is None, using shared_state
                )
                proc.start()
                executor_subprocs.append(proc)

            # Wait for all executor processes
            for proc in executor_subprocs:
                proc.join()
        else:
            # Launch P2P server as subprocess (with scheduler)
            p2p_server_process = launch_p2p_server_process(
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
                param_mem_ratio=args.param_mem_ratio,
                kvcache_mem_ratio=args.kvcache_mem_ratio,
                shared_state=shared_state,
            )

            # Wait for layer allocation from scheduler (via shared state)
            logger.debug("Waiting for layer allocation from scheduler...")
            max_wait_time = 300  # 5 minutes
            wait_start = time.time()
            while (
                shared_state.get("block_start_index") is None
                or shared_state.get("block_end_index") is None
                or shared_state.get("model_name") is None
            ):
                if time.time() - wait_start > max_wait_time:
                    logger.error("Timeout waiting for layer allocation from scheduler")
                    raise RuntimeError("Failed to get layer allocation from scheduler")
                time.sleep(1)

            # Get layer allocation from shared state
            args.start_layer = shared_state["block_start_index"]
            args.end_layer = shared_state["block_end_index"]
            args.model_path = shared_state["model_name"]
            args.tp_size = shared_state.get("tp_size", args.tp_size)

            logger.debug(
                f"Start Executor with start_layer: {args.start_layer}, end_layer: {args.end_layer}, "
                f"model: {args.model_path}"
            )

            if args.log_level != "DEBUG":
                display_parallax_join(args.model_path)
            check_latest_release()

            # only launch http server on head node
            if args.start_layer == 0:
                http_server_process = launch_http_server(args)

            # Main execution loop with layer reallocation support
            while True:
                try:
                    # Launch all executor processes (including tp_rank=0)
                    executor_subprocs = []
                    for tp_rank in range(args.tp_size):
                        args_copy = argparse.Namespace(**vars(args))
                        args_copy.tp_rank = tp_rank
                        proc = multiprocessing.Process(
                            target=run_executor_process,
                            args=(
                                args_copy,
                                None,
                                shared_state,
                            ),  # gradient_server is None, using shared_state
                        )
                        proc.start()
                        executor_subprocs.append(proc)

                    # Wait a bit for executors to initialize, then mark as ready
                    # This allows P2P server to report is_active=True
                    time.sleep(2)  # Give executors time to start
                    shared_state["status"] = ServerState.READY.value

                    # Monitor executor processes and check for layer allocation changes
                    # Use timeout-based join to periodically check shared state
                    layer_reallocation_detected = False
                    while any(proc.is_alive() for proc in executor_subprocs):
                        # Wait for processes with timeout to allow periodic checking
                        for proc in executor_subprocs:
                            if proc.is_alive():
                                proc.join(timeout=1.0)  # Check every 1 second

                        # Check if layer allocation changed while executor is running
                        if shared_state.get("_layer_allocation_changed", False):
                            logger.warning(
                                "Layer allocation changed detected! Stopping executor processes to reload..."
                            )

                            # Shutdown all executor processes
                            for executor_process in executor_subprocs:
                                if executor_process.is_alive():
                                    logger.debug(
                                        f"Terminating executor process {executor_process.pid}"
                                    )
                                    stop_executor_process(executor_process)

                            # Update args with new layer allocation from shared state
                            args.start_layer = shared_state["block_start_index"]
                            args.end_layer = shared_state["block_end_index"]
                            if shared_state.get("model_name"):
                                args.model_path = shared_state["model_name"]
                            if shared_state.get("tp_size"):
                                args.tp_size = shared_state["tp_size"]

                            # Reset the flag and set status to INITIALIZING
                            shared_state["_layer_allocation_changed"] = False
                            shared_state["status"] = ServerState.INITIALIZING.value

                            logger.info(
                                f"Creating new executor with layers [{args.start_layer}, {args.end_layer})"
                            )
                            layer_reallocation_detected = True
                            break  # Exit inner loop to restart executor processes

                    # Check if we exited due to layer allocation change
                    if layer_reallocation_detected:
                        # Continue outer loop to restart executor processes with new layer allocation
                        continue  # Create new executor in next iteration
                    elif shared_state.get("_layer_allocation_changed", False):
                        # This should not happen as we check above, but handle it anyway
                        logger.warning("Layer allocation changed after executor exit, reloading...")
                        args.start_layer = shared_state["block_start_index"]
                        args.end_layer = shared_state["block_end_index"]
                        if shared_state.get("model_name"):
                            args.model_path = shared_state["model_name"]
                        if shared_state.get("tp_size"):
                            args.tp_size = shared_state["tp_size"]
                        shared_state["_layer_allocation_changed"] = False
                        continue  # Create new executor in next iteration
                    else:
                        # All processes exited normally
                        break  # Normal exit
                except KeyboardInterrupt:
                    logger.debug("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    logger.exception(f"Executor error: {e}")
                    # Shutdown all executor processes on error
                    for proc in executor_subprocs:
                        if proc.is_alive():
                            stop_executor_process(proc)
                    raise
    except KeyboardInterrupt:
        logger.debug("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(e)
    finally:
        # Shutdown all processes
        logger.debug("Shutting down all processes...")

        # Shutdown executor subprocesses
        for executor_process in executor_subprocs:
            if executor_process.is_alive():
                stop_executor_process(executor_process)

        # Shutdown P2P server subprocess
        if p2p_server_process is not None:
            stop_p2p_server(p2p_server_process)

        # Shutdown http server
        if http_server_process is not None:
            stop_http_server(http_server_process)

        logger.debug("All processes shut down.")
