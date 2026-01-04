import os
import random
import time

from lattica import Lattica

from parallax.cli import (
    ENDPOINT_PROTOCOL_VERSION,
    PUBLIC_INITIAL_PEERS,
    PUBLIC_RELAY_SERVERS,
)
from parallax.utils.weight_refit_utils import release_disk_storage
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class DataTransferHandler:
    def __init__(self):
        self.scheduler_manage = None
        self.endpoint_lattica = None
        self.initial_peers = PUBLIC_INITIAL_PEERS
        self.relay_servers = PUBLIC_RELAY_SERVERS
        self.protocol_version = ENDPOINT_PROTOCOL_VERSION
        self.host_maddrs = [
            f"/ip4/0.0.0.0/tcp/0",
            f"/ip4/0.0.0.0/udp/0/quic-v1",
        ]
        self.block_path = "/tmp/endpoint"  # workaround to distinct with worker node path
        self.request_data = {}
        self._start_endpoint_lattica()

    def _start_endpoint_lattica(self):
        """
        Start another lattica instance data transfer within endpoints and/or with echo.
        """
        # Reuse existing Lattica if running
        if self.endpoint_lattica is not None:
            logger.debug("Endpoint lattica already running, reusing existing instance")
            return

        self.endpoint_lattica = (
            Lattica.builder().with_listen_addrs(self.host_maddrs).with_key_path(".")
        )

        if len(self.relay_servers) > 0:
            logger.info(f"Endpoint using relay servers: {self.relay_servers}")
            logger.info(f"Endpoint launched with protocol version: {self.protocol_version}")
            self.endpoint_lattica.with_relay_servers(self.relay_servers).with_dcutr(
                True
            ).with_protocol(self.protocol_version)

        if len(self.initial_peers) > 0:
            logger.info(f"Endpoint using initial peers: {self.initial_peers}")
            self.endpoint_lattica.with_bootstraps(self.initial_peers)

        folder = os.path.exists(self.block_path)
        if not folder:
            os.makedirs(self.block_path)
        self.endpoint_lattica.with_storage_path(self.block_path)
        self.endpoint_lattica.with_dht_db_path(self.block_path)
        self.endpoint_lattica.with_key_path(self.block_path)

        self.endpoint_lattica.build()
        logger.debug("Endpoint lattica built")

    def _get_full_weight_blocks(self, message):

        def _download_weight(cid):
            raw_data = None
            time_out = 10 * 60  # 10 minutes timeout
            time_begin_get_block = time.time()
            time_end_get_block = None
            peer_id = None
            while True:
                try:
                    cur_time = time.time()
                    if cur_time - time_begin_get_block > time_out:
                        logger.warning(f"Failed to get_block after 10 minutes! cid={cid}")
                        return None
                    peer_id, raw_data = self.endpoint_lattica.get_block(cid, timeout_secs=30)
                    time_end_get_block = time.time()
                    break
                except Exception:
                    logger.warning(f"Failed to get block: {cid}. Retry in 1 second.")
                    time.sleep(1)
            if raw_data is None:
                return None
            interval_get_block = time_end_get_block - time_begin_get_block
            logger.info(
                f"Finish download cid={cid}, get_block={interval_get_block}s, peer_id={peer_id}"
            )
            new_cid = self.scheduler_manage.lattica.put_block(raw_data)
            return new_cid

        release_disk_storage(["/tmp/endpoint", "/tmp/scheduler"])

        cid_list = message.get("cid", None)
        random.seed(time.time())
        random.shuffle(cid_list)
        new_cid_list = []

        # sleep 30s for lattica direct connection
        time.sleep(30)

        while True:
            if len(cid_list) == 0:
                break
            else:
                cid = cid_list.pop()
                logger.info(f"Start downloading refit weight {cid}")
                new_cid = _download_weight(cid)
                if new_cid is None:
                    logger.warning(f"Endpoint failed to download full weight")
                    break
                else:
                    new_cid_list.append(new_cid)
        return new_cid_list

    def set_scheduler_manage(self, scheduler_manage):
        self.scheduler_manage = scheduler_manage

    def run(self):
        while True:
            refit_data = self.scheduler_manage.refit_data
            if len(refit_data) > 0:
                new_cid_list = self._get_full_weight_blocks(refit_data)
                # replace cid_list and feed to scheduler
                refit_data["cid"] = new_cid_list
                self.scheduler_manage.scheduler.refit_request = refit_data
                self.scheduler_manage.scheduler.refit_set = set()
                self.scheduler_manage.refit_data = {}
            time.sleep(1)
