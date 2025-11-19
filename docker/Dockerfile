FROM lmsysorg/sglang:v0.5.5

ENV SGLANG_ENABLE_JIT_DEEPGEMM=0

WORKDIR /parallax

COPY README.md ./README.md
COPY src ./src
COPY pyproject.toml ./pyproject.toml

RUN pip install -e '.[gpu]'
