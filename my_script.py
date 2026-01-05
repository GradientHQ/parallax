import mlx.core as mx
import time

world = mx.distributed.init(backend="jaccl")
x = mx.distributed.all_sum(mx.ones(10))
print(world.rank(), x)