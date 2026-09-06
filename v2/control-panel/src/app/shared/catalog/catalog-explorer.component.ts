import { CommonModule } from '@angular/common';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  AdminCityList,
  AdminCitySummary,
  AdminPoiDetail,
  AdminPoiPage,
  AdminPoiSummary,
  BootstrapResult,
  PoiTypeOption,
  PoiUpdateRequest,
} from '../../core/admin-catalog.model';
import { PoiMapPoint } from '../../core/admin-overview.model';
import { PoiMapComponent } from '../maps/poi-map.component';
import { PoiCallTestComponent } from './poi-call-test.component';

@Component({
  selector: 'locus-catalog-explorer',
  standalone: true,
  imports: [CommonModule, FormsModule, PoiCallTestComponent, PoiMapComponent],
  templateUrl: './catalog-explorer.component.html',
  styleUrl: './catalog-explorer.component.scss',
})
export class CatalogExplorerComponent {
  private readonly http = inject(HttpClient);
  private readonly pageSize = 100;

  readonly cities = signal<AdminCitySummary[]>([]);
  readonly cityTotal = signal(0);
  readonly selectedCity = signal<AdminCitySummary | null>(null);
  readonly pois = signal<AdminPoiSummary[]>([]);
  readonly poiTotal = signal(0);
  readonly selectedPoi = signal<AdminPoiDetail | null>(null);
  readonly loadingCities = signal(true);
  readonly loadingPois = signal(false);
  readonly detailLoading = signal(false);
  readonly error = signal<string | null>(null);
  readonly viewMode = signal<'map' | 'list'>('map');
  readonly mapPoints = computed<PoiMapPoint[]>(() => {
    const city = this.selectedCity();
    const cityLat = city?.lat === null || city?.lat === undefined ? null : Number(city.lat);
    const cityLng = city?.lng === null || city?.lng === undefined ? null : Number(city.lng);
    return this.pois().flatMap((poi) => {
      if (poi.lat === null || poi.lng === null) return [];
      const lat = Number(poi.lat);
      const lng = Number(poi.lng);
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) return [];
      if (cityLat !== null && cityLng !== null && this.distanceKm(cityLat, cityLng, lat, lng) > 100) return [];
      return [{
        id: poi.public_id,
        name: poi.name,
        city: city?.name ?? '',
        lat,
        lng,
      }];
    });
  });
  readonly mapCenter = computed(() => {
    const city = this.selectedCity();
    if (!city || city.lat === null || city.lng === null) return null;
    const lat = Number(city.lat);
    const lng = Number(city.lng);
    return Number.isFinite(lat) && Number.isFinite(lng) ? { lat, lng } : null;
  });

  readonly seedPanelOpen = signal(false);
  readonly seeding = signal(false);
  readonly seedResult = signal<BootstrapResult | null>(null);
  readonly seedError = signal<string | null>(null);

  readonly editingPoi = signal(false);
  readonly savingPoi = signal(false);
  readonly editError = signal<string | null>(null);
  readonly poiTypes = signal<PoiTypeOption[]>([]);
  editForm = this.blankEditForm();
  editNames: { lang: string; value: string }[] = [];
  editShortDescriptions: { lang: string; value: string }[] = [];

  cityQuery = '';
  poiQuery = '';
  active = '';
  offset = 0;
  seedLat: number | null = null;
  seedLng: number | null = null;
  seedRadiusKm = 3;
  seedLimit = 40;
  seedUseAi = false;

  constructor() {
    this.loadCities();
  }

  loadCities(selectPublicId?: string): void {
    this.loadingCities.set(true);
    const params = new HttpParams().set('q', this.cityQuery.trim()).set('limit', 300);
    this.http.get<AdminCityList>('/admin/v2/catalog/cities', { params }).subscribe({
      next: (result) => {
        this.cities.set(result.items);
        this.cityTotal.set(result.total);
        this.loadingCities.set(false);
        const toSelect = selectPublicId
          ? result.items.find((city) => city.public_id === selectPublicId)
          : null;
        if (toSelect) {
          this.selectCity(toSelect);
        } else if (!this.selectedCity() && result.items.length) {
          this.selectCity(result.items[0]);
        }
      },
      error: () => {
        this.error.set('No hemos podido abrir el catálogo de ciudades.');
        this.loadingCities.set(false);
      },
    });
  }

  selectCity(city: AdminCitySummary): void {
    this.selectedCity.set(city);
    this.pois.set([]);
    this.poiTotal.set(0);
    this.offset = 0;
    this.loadPois();
  }

  loadPois(): void {
    this.loadingPois.set(true);
    this.error.set(null);
    let params = new HttpParams()
      .set('q', this.poiQuery.trim())
      .set('limit', this.pageSize)
      .set('offset', this.offset);
    const city = this.selectedCity();
    if (city) params = params.set('city_id', city.id);
    if (this.active) params = params.set('active', this.active);
    this.http.get<AdminPoiPage>('/admin/v2/catalog/pois', { params }).subscribe({
      next: (result) => {
        this.pois.set(result.items);
        this.poiTotal.set(result.total);
        this.loadingPois.set(false);
      },
      error: () => {
        this.error.set('No hemos podido leer los lugares de esta ciudad.');
        this.loadingPois.set(false);
      },
    });
  }

  openPoi(poi: AdminPoiSummary): void {
    this.detailLoading.set(true);
    this.http.get<AdminPoiDetail>(`/admin/v2/catalog/pois/${poi.id}`).subscribe({
      next: (detail) => {
        this.selectedPoi.set(detail);
        this.detailLoading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido abrir la ficha del lugar.');
        this.detailLoading.set(false);
      },
    });
  }

  openPoiByPublicId(publicId: string): void {
    const poi = this.pois().find((candidate) => candidate.public_id === publicId);
    if (poi) this.openPoi(poi);
  }

  closePoi(): void {
    this.selectedPoi.set(null);
    this.editingPoi.set(false);
  }

  startEdit(): void {
    const poi = this.selectedPoi();
    if (!poi) return;
    if (!this.poiTypes().length) this.loadPoiTypes();
    this.editForm = {
      name: poi.name,
      short_description: poi.short_description,
      long_description: poi.long_description,
      lat: poi.lat === null ? '' : String(poi.lat),
      lng: poi.lng === null ? '' : String(poi.lng),
      poi_type_code: poi.type_code ?? '',
      is_active: poi.is_active,
      wikidata_id: poi.wikidata_id,
      wikipedia_title: poi.wikipedia_title,
      google_place_id: poi.google_place_id,
    };
    this.editNames = Object.entries(poi.names).map(([lang, value]) => ({ lang, value }));
    this.editShortDescriptions = Object.entries(poi.short_descriptions).map(([lang, value]) => ({ lang, value }));
    this.editError.set(null);
    this.editingPoi.set(true);
  }

  cancelEdit(): void {
    this.editingPoi.set(false);
    this.editError.set(null);
  }

  loadPoiTypes(): void {
    this.http.get<PoiTypeOption[]>('/admin/v2/catalog/poi-types').subscribe({
      next: (types) => this.poiTypes.set(types),
      error: () => this.poiTypes.set([]),
    });
  }

  addNameRow(): void {
    this.editNames = [...this.editNames, { lang: '', value: '' }];
  }

  removeNameRow(index: number): void {
    this.editNames = this.editNames.filter((_, i) => i !== index);
  }

  addDescriptionRow(): void {
    this.editShortDescriptions = [...this.editShortDescriptions, { lang: '', value: '' }];
  }

  removeDescriptionRow(index: number): void {
    this.editShortDescriptions = this.editShortDescriptions.filter((_, i) => i !== index);
  }

  saveEdit(): void {
    const poi = this.selectedPoi();
    if (!poi) return;
    this.savingPoi.set(true);
    this.editError.set(null);

    const payload: PoiUpdateRequest = {
      name: this.editForm.name,
      short_description: this.editForm.short_description,
      long_description: this.editForm.long_description,
      lat: this.editForm.lat,
      lng: this.editForm.lng,
      poi_type_code: this.editForm.poi_type_code || null,
      is_active: this.editForm.is_active,
      wikidata_id: this.editForm.wikidata_id,
      wikipedia_title: this.editForm.wikipedia_title,
      google_place_id: this.editForm.google_place_id,
      names: this.rowsToRecord(this.editNames),
      short_descriptions: this.rowsToRecord(this.editShortDescriptions),
    };

    this.http.put<AdminPoiDetail>(`/admin/v2/catalog/pois/${poi.id}`, payload).subscribe({
      next: (updated) => {
        this.selectedPoi.set(updated);
        this.pois.update((items) =>
          items.map((item) => (item.id === updated.id ? { ...item, ...updated } : item)),
        );
        this.savingPoi.set(false);
        this.editingPoi.set(false);
      },
      error: ({ error }) => {
        this.editError.set(error?.detail ?? 'No se ha podido guardar el POI.');
        this.savingPoi.set(false);
      },
    });
  }

  private rowsToRecord(rows: { lang: string; value: string }[]): Record<string, string> {
    const record: Record<string, string> = {};
    for (const row of rows) {
      const lang = row.lang.trim().toLowerCase();
      if (lang && row.value.trim()) record[lang] = row.value.trim();
    }
    return record;
  }

  private blankEditForm() {
    return {
      name: '',
      short_description: '',
      long_description: '',
      lat: '',
      lng: '',
      poi_type_code: '',
      is_active: true,
      wikidata_id: '',
      wikipedia_title: '',
      google_place_id: '',
    };
  }

  previous(): void {
    this.offset = Math.max(0, this.offset - this.pageSize);
    this.loadPois();
  }

  next(): void {
    if (this.offset + this.pageSize >= this.poiTotal()) return;
    this.offset += this.pageSize;
    this.loadPois();
  }

  toggleSeedPanel(): void {
    this.seedPanelOpen.set(!this.seedPanelOpen());
    this.seedError.set(null);
  }

  onMapClicked(point: { lat: number; lng: number }): void {
    if (!this.seedPanelOpen()) return;
    this.seedLat = Math.round(point.lat * 1e6) / 1e6;
    this.seedLng = Math.round(point.lng * 1e6) / 1e6;
  }

  runSeed(): void {
    if (this.seedLat === null || this.seedLng === null) {
      this.seedError.set('Fija primero un punto: clica el mapa o escribe lat/lng.');
      return;
    }
    this.seeding.set(true);
    this.seedError.set(null);
    this.seedResult.set(null);
    this.http.post<BootstrapResult>('/admin/v2/catalog/bootstrap-from-location', {
      lat: this.seedLat,
      lng: this.seedLng,
      radius_km: this.seedRadiusKm,
      limit: this.seedLimit,
      use_ai_candidates: this.seedUseAi,
    }).subscribe({
      next: (result) => {
        this.seedResult.set(result);
        this.seeding.set(false);
        this.loadCities(result.city_public_id);
      },
      error: ({ error }) => {
        this.seedError.set(error?.detail ?? 'El sembrado no ha podido completarse.');
        this.seeding.set(false);
      },
    });
  }

  closeSeedResult(): void {
    this.seedResult.set(null);
  }

  formattedMetadata(poi: AdminPoiDetail): string {
    return JSON.stringify(poi.metadata, null, 2);
  }

  private distanceKm(latA: number, lngA: number, latB: number, lngB: number): number {
    const toRadians = (value: number) => value * Math.PI / 180;
    const latDelta = toRadians(latB - latA);
    const lngDelta = toRadians(lngB - lngA);
    const a = Math.sin(latDelta / 2) ** 2
      + Math.cos(toRadians(latA)) * Math.cos(toRadians(latB)) * Math.sin(lngDelta / 2) ** 2;
    return 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }
}
