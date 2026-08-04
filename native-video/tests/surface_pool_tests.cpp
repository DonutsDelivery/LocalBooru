// SPDX-License-Identifier: MIT
#include "surface_pool.h"

#include <cassert>
#include <cstdint>

using namespace localbooru::native_video;

int main() {
  SurfacePool pool(3);
  pool.configure(7);
  assert(pool.capacity() == 3);
  assert(pool.available() == 3);

  const auto first_id = pool.acquire_for_producer();
  assert(first_id && *first_id == 0);
  const auto first = pool.publish(*first_id);
  assert(first && first->generation == 7 && first->sequence == 1);
  assert(pool.consumer_owned() == 1);

  const auto second_id = pool.acquire_for_producer();
  const auto third_id = pool.acquire_for_producer();
  assert(second_id && third_id);
  const auto second = pool.publish(*second_id);
  const auto third = pool.publish(*third_id);
  assert(second && third);
  assert(!pool.acquire_for_producer());

  FrameLease stale = *first;
  ++stale.sequence;
  assert(!pool.release(stale));
  assert(!pool.acquire_for_producer());
  assert(pool.release(*first));
  assert(pool.available() == 1);
  assert(pool.acquire_for_producer() == first->buffer_id);

  pool.configure(8);
  assert(pool.available() == 3);
  assert(pool.consumer_owned() == 0);
  assert(!pool.release(*second));

  const auto cancelled_id = pool.acquire_for_producer();
  assert(cancelled_id);
  assert(pool.cancel_producer(*cancelled_id));
  assert(!pool.cancel_producer(*cancelled_id));
  assert(pool.available() == 3);

  const auto replacement_id = pool.acquire_for_producer();
  const auto replacement = pool.publish(*replacement_id);
  assert(replacement && replacement->generation == 8);
  assert(replacement->sequence > third->sequence);
  return 0;
}
