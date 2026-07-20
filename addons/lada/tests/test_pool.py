from localbooru_lada.pool import FramePool, PoolExhausted, StaleLease


def test_pool_never_exceeds_three_in_flight_buffers():
    with FramePool(buffer_count=3, buffer_capacity=32) as pool:
        leases = [pool.acquire(generation=4) for _ in range(3)]

        assert pool.in_flight == 3
        try:
            pool.acquire(generation=4, block=False)
        except PoolExhausted:
            pass
        else:
            raise AssertionError("fourth lease must be backpressured")

        pool.release(leases[0].buffer_id, leases[0].sequence, generation=4)
        replacement = pool.acquire(generation=4, block=False)
        assert replacement.buffer_id == leases[0].buffer_id
        assert pool.in_flight == 3


def test_seek_retires_old_pool_and_uses_fresh_generation_buffers():
    old_pool = FramePool(buffer_count=3, buffer_capacity=8)
    lease = old_pool.acquire(generation=2)
    old_pool.close()

    try:
        old_pool.release(lease.buffer_id, lease.sequence, generation=2)
    except RuntimeError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("a retired pool must reject old-generation releases")

    with FramePool(buffer_count=3, buffer_capacity=8) as new_pool:
        assert new_pool.acquire(generation=3, block=False).generation == 3


def test_duplicate_release_is_rejected():
    with FramePool(buffer_count=1, buffer_capacity=8) as pool:
        lease = pool.acquire(generation=1)
        pool.release(lease.buffer_id, lease.sequence, generation=1)

        try:
            pool.release(lease.buffer_id, lease.sequence, generation=1)
        except StaleLease:
            pass
        else:
            raise AssertionError("duplicate release must be rejected")
