import datetime
import jax
import random
import string


def simple_timeit(f, *args, tries=10, task=None, enable_profile=False):
  """Simple utility to time a function for multiple runs"""
  assert task is not None

  trace_name = f"{task}"  # + '_' ]+ ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  trace_dir = f"/tmp/{trace_name}"
  print(trace_dir)

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  if enable_profile:
    jax.profiler.start_trace(trace_dir)
  for _ in range(tries):
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())
  if enable_profile:
    jax.profiler.stop_trace()
  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  print(f"Average time ms for mm for {task} is {round(average_time_ms, 3)}")
  return average_time_ms / 1000


def simple_timeit(f, *args, tries=10, task=None, enable_profile=False):
  """Simple utility to time a function for multiple runs"""
  assert task is not None

  trace_name = f"t_{task}_" + "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  trace_dir = f"/tmp/{trace_name}"

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  if enable_profile:
    jax.profiler.start_trace(trace_dir)

  for _ in range(tries):
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())
  if enable_profile:
    jax.profiler.stop_trace()

  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)

  print(f"{task}: average time milliseconds: {average_time_ms:.2f}")
  if enable_profile:
    print(f"trace {trace_dir}")
  return average_time_ms
