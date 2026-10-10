"""Borrar la cuenta: que no quede nada que identifique a la persona."""

from locus_v2.identity.application.account_deletion import anonymize
from locus_v2.identity.models import User, UserStatus


def test_nothing_identifying_is_left_and_the_account_is_closed() -> None:
    user = User(
        public_id="abc", email="dizz01@gmail.com", display_name="Carlos García",
        preferred_name="Carlos", avatar_url="https://lh3/foto", provider_subject="google-123",
        profile_context="Viajo con mis padres", status=UserStatus.ACTIVE,
    )
    anonymize(user)

    datos = " ".join(
        str(v) for v in (
            user.email, user.display_name, user.preferred_name, user.avatar_url,
            user.provider_subject, user.profile_context,
        )
    )
    for rastro in ("dizz01", "Carlos", "lh3", "google-123", "padres"):
        assert rastro not in datos
    assert user.status == UserStatus.DELETED
    # Un email único, para que la misma persona pueda volver con una cuenta nueva.
    assert user.email == "deleted-abc@deleted.locusguide.es"
