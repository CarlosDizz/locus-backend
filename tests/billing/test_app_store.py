"""Una compra del iPhone solo cuenta si viene firmada por Apple."""

from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from locus_v2.billing.infrastructure.app_store import (
    AppStoreVerificationError,
    AppStoreVerifier,
)

SETTINGS = SimpleNamespace(app_store_bundle_id="com.carlos.locusia", app_store_app_apple_id=1)


def test_garbage_is_not_a_purchase() -> None:
    with pytest.raises(AppStoreVerificationError):
        AppStoreVerifier(SETTINGS).verify("no-es-un-jws")


def test_a_purchase_signed_by_someone_else_is_rejected() -> None:
    """Firmada con una clave cualquiera, sin la cadena de Apple: no vale."""
    key = ec.generate_private_key(ec.SECP256R1())
    falsa = jwt.encode(
        {
            "bundleId": "com.carlos.locusia", "productId": "locus_ios_top_up_20",
            "transactionId": "1", "environment": "Production",
        },
        key,
        algorithm="ES256",
        headers={"x5c": ["MIIB", "MIIB", "MIIB"]},
    )
    with pytest.raises(AppStoreVerificationError):
        AppStoreVerifier(SETTINGS).verify(falsa)
