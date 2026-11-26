"""Optional production observers. No dependency on the web service or test fixtures."""
from contextlib import contextmanager
import hashlib
import time
from uuid import uuid4


class StopRequested(Exception):
    """Stop at a window boundary; the last published checkpoint remains valid."""


def emit(observer, stage, state, **data):
    if observer is not None:
        observer.event(stage, state, **data)


def visible_prompt(prompt):
    """Keep visible text; don't duplicate megabytes of embedded image bytes in logs."""
    if isinstance(prompt, str):
        return prompt
    result = []
    for item in prompt:
        if item.get('type') == 'image_url':
            value = item['image_url']['url']
            result.append(dict(type='image', sha256=hashlib.sha256(value.encode()).hexdigest(),
                               encoded_characters=len(value)))
        else:
            result.append(item)
    return result


@contextmanager
def stage(observer, name, **data):
    attempt = uuid4().hex
    started = time.monotonic()
    emit(observer, name, 'started', attempt_id=attempt, **data)
    try:
        yield
    except Exception as exc:
        emit(observer, name, 'failed', attempt_id=attempt, elapsed=time.monotonic()-started,
             error=f'{type(exc).__name__}: {exc}')
        raise
    else:
        emit(observer, name, 'completed', attempt_id=attempt, elapsed=time.monotonic()-started)
