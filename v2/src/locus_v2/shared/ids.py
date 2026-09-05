from uuid import uuid4


def new_public_id() -> str:
    return str(uuid4())
