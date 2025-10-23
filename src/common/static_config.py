PUBLIC_INITIAL_PEERS = [
    "/dns4/bootstrap-lattica.gradient.network/udp/18080/quic-v1/p2p/12D3KooWJHXvu8TWkFn6hmSwaxdCLy4ZzFwr4u5mvF9Fe2rMmFXb",
    "/dns4/bootstrap-lattica.gradient.network/tcp/18080/p2p/12D3KooWJHXvu8TWkFn6hmSwaxdCLy4ZzFwr4u5mvF9Fe2rMmFXb",
    "/dns4/bootstrap-lattica-us.gradient.network/udp/18080/quic-v1/p2p/12D3KooWFD8NoyHfmVxLVCocvXJBjwgE9RZ2bgm2p5WAWQax4FoQ",
    "/dns4/bootstrap-lattica-us.gradient.network/tcp/18080/p2p/12D3KooWFD8NoyHfmVxLVCocvXJBjwgE9RZ2bgm2p5WAWQax4FoQ",
    "/dns4/bootstrap-lattica-eu.gradient.network/udp/18080/quic-v1/p2p/12D3KooWCNuEF4ro95VA4Lgq4NvjdWfJFoTcvWsBA7Z6VkBByPtN",
    "/dns4/bootstrap-lattica-eu.gradient.network/tcp/18080/p2p/12D3KooWCNuEF4ro95VA4Lgq4NvjdWfJFoTcvWsBA7Z6VkBByPtN",
]

PUBLIC_RELAY_SERVERS = [
    "/dns4/relay-lattica.gradient.network/udp/18080/quic-v1/p2p/12D3KooWDaqDAsFupYvffBDxjHHuWmEAJE4sMDCXiuZiB8aG8rjf",
    "/dns4/relay-lattica.gradient.network/tcp/18080/p2p/12D3KooWDaqDAsFupYvffBDxjHHuWmEAJE4sMDCXiuZiB8aG8rjf",
    "/dns4/relay-lattica-us.gradient.network/udp/18080/quic-v1/p2p/12D3KooWHMXi6SCfaQzLcFt6Th545EgRt4JNzxqmDeLs1PgGm3LU",
    "/dns4/relay-lattica-us.gradient.network/tcp/18080/p2p/12D3KooWHMXi6SCfaQzLcFt6Th545EgRt4JNzxqmDeLs1PgGm3LU",
    "/dns4/relay-lattica-eu.gradient.network/udp/18080/quic-v1/p2p/12D3KooWRAuR7rMNA7Yd4S1vgKS6akiJfQoRNNexTtzWxYPiWfG5",
    "/dns4/relay-lattica-eu.gradient.network/tcp/18080/p2p/12D3KooWRAuR7rMNA7Yd4S1vgKS6akiJfQoRNNexTtzWxYPiWfG5",
]


def get_relay_params():
    return [
        "--relay-servers",
        *PUBLIC_RELAY_SERVERS,
        "--initial-peers",
        *PUBLIC_INITIAL_PEERS,
    ]
