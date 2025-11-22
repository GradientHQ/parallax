import logging

logger = logging.getLogger(__name__)


# def set_global_expert_location_metadata(value):
#     global _global_expert_location_metadata
#     assert _global_expert_location_metadata is None
#     _global_expert_location_metadata = value


def apply_expert_location_monkey_patch():

    import sglang.srt.eplb.expert_location as expert_location

    def patched_set_global_expert_location_metadata(metadata):
        expert_location._global_expert_location_metadata = metadata
        logger.debug("Set global expert location metadata (patched, no assert)")

    expert_location.set_global_expert_location_metadata = (
        patched_set_global_expert_location_metadata
    )
