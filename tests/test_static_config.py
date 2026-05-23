from backend.server.static_config import MODELS


def test_glm_5_1_uses_mlx_community_model():
    assert MODELS["zai-org/GLM-5.1"] == "mlx-community/GLM-5.1"
