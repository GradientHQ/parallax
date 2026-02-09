"""
Launch the vLLM server for RL.
"""

import multiprocessing
import os
import subprocess
import tempfile
import time

from parallax.p2p.server import ServerState, launch_p2p_server_process, stop_p2p_server
from parallax.server.executor.factory import stop_executor_process
from parallax.server.http_server import stop_http_server
from parallax.server.server_args import parse_args
from parallax.utils.shared_state import SharedState
from parallax.utils.utils import initialize_nccl_port
from parallax_utils.ascii_anime import display_parallax_join
from parallax_utils.logging_config import get_logger, set_log_level
from parallax_utils.prepare_adapter import download_adapter_config
from parallax_utils.version_check import check_latest_release

logger = get_logger("parallax.launch")


def _update_args_from_shared_state(args, shared_state: SharedState, force_update: bool):
    """Update args with layer allocation from shared state"""
    model_info = shared_state.get_model_info()
    args.start_layer = model_info["block_start_index"]
    args.end_layer = model_info["block_end_index"]
    if args.model_path is not None and force_update == False:
        # Use local model path first
        pass
    elif model_info["model_name"]:
        # Update model_path if provided
        args.model_path = model_info["model_name"]
        logger.debug(f"Updated model_path to: {args.model_path}")
    else:
        assert False, "Neither scheduler nor worker provides a valid model path!"
    # Update tp_size if provided, otherwise keep current value
    args.tp_size = model_info["tp_size"] or args.tp_size
    # Update weight refit switch
    args.enable_weight_refit = model_info["enable_weight_refit"] or args.enable_weight_refit
    args.weight_refit_mode = model_info["weight_refit_mode"] or args.weight_refit_mode


def _stop_executor_processes(executor_subprocs):
    """Stop all executor processes"""
    for executor_process in executor_subprocs:
        logger.debug(f"Terminating executor process {executor_process.pid}")
        stop_executor_process(executor_process)


def _wait_executors_check_layer_change(shared_state: SharedState, executor_subprocs):
    """Wait for executor processes and check if layer allocation changed.

    Returns:
        True if layer allocation changed (need to reload executors),
        False if all executors exited normally.
    """
    while True:
        for proc in executor_subprocs:
            if proc.poll() is not None:
                break
            else:
                time.sleep(1)

        if shared_state.get_layer_allocation_changed():
            return True


def _launch_vllm_server(args):
    cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        f"--model={args.model_path}",
        f"--host={args.host}",
        f"--port={args.port}",
        f"--tensor-parallel-size={args.tp_size}",
        f"--gpu-memory-utilization={args.kv_cache_memory_fraction}",
        f"--max-model-len={args.max_sequence_length}",
        f"--enable-lora",
        f"--lora-modules={args.lora_paths[0]}={args.lora_paths[0]}",
        "--trust-remote-code",
    ]
    return subprocess.Popen(cmd)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    p2p_server_process = None
    http_server_process = None
    executor_subprocs = []
    # Shared state for layer allocation info (used when P2P server is in subprocess)
    shared_state = SharedState.create()
    shared_state.set_status(ServerState.JOINING.value)

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

        # Pipe for subprocess communication
        conn_main, conn_refit = multiprocessing.Pipe()

        # Prepare lora config
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"
        lora_path = None
        if args.lora_paths is not None and len(args.lora_paths) > 0:
            lora_path = args.lora_paths[0]
            download_adapter_config(lora_path)

        # Launch P2P server as subprocess (with scheduler)
        # Pass dict to subprocess (multiprocessing requires serializable objects)
        p2p_server_process = launch_p2p_server_process(
            initial_peers=args.initial_peers,
            scheduler_addr=args.scheduler_addr,
            relay_servers=args.relay_servers,
            pp_start_layer=args.start_layer,
            pp_end_layer=args.end_layer,
            hidden_layers=None,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
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
            shared_state=shared_state.dict,  # Pass dict to subprocess
            log_level=args.log_level,
            lora_path=lora_path,
            conn=conn_main,
        )

        # Wait for layer allocation from scheduler (via shared state)
        logger.debug("Waiting for layer allocation from scheduler...")
        max_wait_time = 300  # 5 minutes
        wait_start = time.time()
        while True:
            model_info = shared_state.get_model_info()
            if (
                model_info["block_start_index"] is not None
                and model_info["block_end_index"] is not None
                and model_info["model_name"] is not None
            ):
                break
            if time.time() - wait_start > max_wait_time:
                logger.error("Timeout waiting for layer allocation from scheduler")
                raise RuntimeError("Failed to get layer allocation from scheduler")
            time.sleep(1)

        # Get layer allocation from shared state
        _update_args_from_shared_state(args, shared_state, force_update=False)

        logger.debug(
            f"Start Executor with start_layer: {args.start_layer}, end_layer: {args.end_layer}, "
            f"model: {args.model_path}"
        )

        if args.log_level != "DEBUG":
            display_parallax_join(args.model_path)
        check_latest_release()

        # Main execution loop with layer reallocation support
        while True:
            try:
                proc = _launch_vllm_server(args)
                executor_subprocs.append(proc)
                shared_state.set_status(ServerState.READY.value)

                # Wait for executors and restart if layer allocation changes
                if _wait_executors_check_layer_change(shared_state, executor_subprocs):
                    logger.warning("Layer allocation changed! Stopping executors to reload...")
                    # Reset flag and set status to INITIALIZING
                    shared_state.update(
                        _layer_allocation_changed=False,
                        status=ServerState.INITIALIZING.value,
                    )
                    _stop_executor_processes(executor_subprocs)
                    if http_server_process is not None:
                        stop_http_server(http_server_process)
                    _update_args_from_shared_state(args, shared_state, force_update=True)
                    logger.info(
                        f"Reloading executor with layers [{args.start_layer}, {args.end_layer})"
                    )
                    continue

                # All processes exited normally
                break
            except KeyboardInterrupt:
                logger.debug("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.exception(f"Executor error: {e}")
                # Shutdown all executor processes on error
                for proc in executor_subprocs:
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
            stop_executor_process(executor_process)

        # Shutdown P2P server subprocess
        if p2p_server_process is not None:
            stop_p2p_server(p2p_server_process)

        # Shutdown http server
        if http_server_process is not None:
            stop_http_server(http_server_process)

        logger.debug("All processes shut down.")
