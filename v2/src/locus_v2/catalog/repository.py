from locus_v2.catalog.models import City, Poi
from locus_v2.kernel.infrastructure.sqlalchemy_repository import SQLAlchemyRepository


class CityRepository(SQLAlchemyRepository[City]):
    model = City


class PoiRepository(SQLAlchemyRepository[Poi]):
    model = Poi
