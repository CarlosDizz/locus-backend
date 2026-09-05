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
} from '../../core/admin-catalog.model';
import { PoiMapPoint } from '../../core/admin-overview.model';
import { PoiMapComponent } from '../maps/poi-map.component';

@Component({
  selector: 'locus-catalog-explorer',
  standalone: true,
  imports: [CommonModule, FormsModule, PoiMapComponent],
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

  cityQuery = '';
  poiQuery = '';
  active = '';
  offset = 0;

  constructor() {
    this.loadCities();
  }

  loadCities(): void {
    this.loadingCities.set(true);
    const params = new HttpParams().set('q', this.cityQuery.trim()).set('limit', 300);
    this.http.get<AdminCityList>('/admin/v2/catalog/cities', { params }).subscribe({
      next: (result) => {
        this.cities.set(result.items);
        this.cityTotal.set(result.total);
        this.loadingCities.set(false);
        if (!this.selectedCity() && result.items.length) this.selectCity(result.items[0]);
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
