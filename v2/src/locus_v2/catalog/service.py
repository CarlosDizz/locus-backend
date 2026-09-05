from locus_v2.catalog.models import City, Poi
from locus_v2.kernel.application.crud import CrudService


class CityService(CrudService[City]):
    pass


class PoiService(CrudService[Poi]):
    pass
