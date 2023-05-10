import simpy

from simpy.resources.resource import Request
from simpy.events import Event

environment = simpy.Environment()

def print_stats(res):
    print(f'{res.count} of {res.capacity} slots are allocated.')
    print(f'  Users: {res.users}')
    print(f'  Queued events: {res.queue}')


def dummy(env: simpy.Environment):
    e = Event(env)
    e.callbacks = [print]
    print("running", env.now, e.processed, e.triggered)
    yield simpy.Timeout(env, delay=3)
    print("fin", env.now, e.processed, e.triggered)

def sub(env):
    event = env.timeout(1)
    print("running sub", env.now, event.processed, event.triggered)
    yield event
    print("fin sub", env.now, event.processed, event.triggered)
    event2 = env.timeout(1)
    print("running sub2", env.now, event.processed, event.triggered)
    yield event2
    print("fin sub2", env.now, event.processed, event.triggered)

def parent(env):
    event = env.timeout(4)
    print("running parent", env.now, event.processed, event.triggered)
    yield event
    print("fin parent", env.now, event.processed, event.triggered)
    # yield env.process(sub(env))
    event2 = env.timeout(4)
    print("running parent2", env.now, event2.processed, event2.triggered)
    yield event2
    print("fin parent2", env.now, event2.processed, event2.triggered)

# environment.process(dummy(environment))
# environment.process(dummy(environment))
# environment.process(dummy(environment))
environment.process(parent(environment))
environment.run(until=10)
