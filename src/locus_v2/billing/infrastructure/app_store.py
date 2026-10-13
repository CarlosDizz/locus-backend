"""Verifica las compras de la app de iPhone (StoreKit 2).

Cada compra llega como un JWS firmado por Apple: la firma se encadena hasta el
certificado raíz de Apple (Apple Root CA - G3, en certs/, huella SHA-256
63:34:3A:BF:…:91:79). Se usa la librería oficial de Apple para no reimplementar
la verificación. No hace falta red: todo va en la propia compra.

Se aceptan también las compras del entorno de pruebas (Sandbox): son las de
TestFlight y, sobre todo, las del revisor de Apple, que revisa la app pagando
en Sandbox. Si se rechazaran, rechazarían la app.
"""

from dataclasses import dataclass
from importlib.resources import files

from appstoreserverlibrary.models.Environment import Environment
from appstoreserverlibrary.signed_data_verifier import (
    SignedDataVerifier,
    VerificationException,
)

from locus_v2.config import Settings


class AppStoreVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AppStorePurchase:
    transaction_id: str
    original_transaction_id: str
    product_id: str
    environment: str
    app_account_token: str | None
    revoked: bool


def _root_certificate() -> bytes:
    return (files("locus_v2.billing.infrastructure") / "certs" / "AppleRootCA-G3.cer").read_bytes()


class AppStoreVerifier:
    def __init__(self, settings: Settings) -> None:
        self.bundle_id = settings.app_store_bundle_id
        self.app_apple_id = settings.app_store_app_apple_id

    def verify(self, signed_transaction: str) -> AppStorePurchase:
        root = [_root_certificate()]
        ultimo_error: Exception | None = None
        for environment in (Environment.PRODUCTION, Environment.SANDBOX):
            verifier = SignedDataVerifier(
                root,
                False,
                environment,
                self.bundle_id,
                self.app_apple_id if environment == Environment.PRODUCTION else None,
            )
            try:
                payload = verifier.verify_and_decode_signed_transaction(signed_transaction)
            except VerificationException as error:
                ultimo_error = error
                continue
            if not payload.transactionId or not payload.productId:
                raise AppStoreVerificationError("La compra de Apple está incompleta")
            return AppStorePurchase(
                transaction_id=str(payload.transactionId),
                original_transaction_id=str(payload.originalTransactionId or ""),
                product_id=str(payload.productId),
                environment=str(payload.rawEnvironment or environment.value),
                app_account_token=str(payload.appAccountToken) if payload.appAccountToken else None,
                revoked=payload.revocationDate is not None,
            )
        raise AppStoreVerificationError(
            "No he podido verificar la compra con Apple"
        ) from ultimo_error
