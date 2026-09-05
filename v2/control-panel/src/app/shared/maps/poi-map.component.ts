import { afterNextRender, Component, effect, ElementRef, input, output, viewChild } from '@angular/core';
import * as L from 'leaflet';

import { PoiMapPoint } from '../../core/admin-overview.model';

@Component({
  selector: 'locus-poi-map',
  standalone: true,
  template: `
    <div class="map-wrap">
      <div #map class="map" aria-label="Mapa de puntos de interés"></div>
      @if (!points().length) {
        <div class="empty"><i></i><b>Sin lugares geolocalizados</b><p>Prueba con otra ciudad o cambia los filtros.</p></div>
      }
    </div>
  `,
  styles: [`
    :host{display:block}.map-wrap{position:relative;height:600px;overflow:hidden;border-top:1px solid var(--locus-line);background:#e6e4d7}.map{width:100%;height:100%;z-index:1}.empty{position:absolute;z-index:2;left:50%;top:50%;transform:translate(-50%,-50%);min-width:260px;padding:28px 34px;background:rgba(255,250,242,.94);text-align:center;box-shadow:var(--locus-shadow);pointer-events:none}.empty i{display:block;width:30px;height:30px;margin:0 auto 14px;border:7px solid var(--locus-blue);border-radius:50%;box-shadow:inset 0 0 0 5px var(--locus-paper);background:var(--locus-terracotta)}.empty b{display:block;font:600 22px "Fraunces",serif}.empty p{color:var(--locus-muted);font-size:10px;margin:7px 0 0}:host ::ng-deep .leaflet-control-zoom a{color:var(--locus-blue);background:var(--locus-paper);border-color:var(--locus-line)}:host ::ng-deep .leaflet-tooltip{border:0;border-radius:4px;background:var(--locus-paper);color:var(--locus-ink);box-shadow:var(--locus-shadow);font:700 10px "Manrope",sans-serif}:host ::ng-deep .leaflet-tooltip-top::before{border-top-color:var(--locus-paper)}@media(max-width:700px){.map-wrap{height:440px}}
  `],
})
export class PoiMapComponent {
  readonly points = input.required<PoiMapPoint[]>();
  readonly cityCenter = input<{ lat: number; lng: number } | null>(null);
  readonly poiSelected = output<string>();
  readonly mapClicked = output<{ lat: number; lng: number }>();
  readonly mapElement = viewChild.required<ElementRef<HTMLDivElement>>('map');

  private map?: L.Map;
  private readonly markers = L.layerGroup();

  constructor() {
    afterNextRender(() => {
      this.map = L.map(this.mapElement().nativeElement, { center: [40.4168, -3.7038], zoom: 5 });
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19,
      }).addTo(this.map);
      this.markers.addTo(this.map);
      this.map.on('click', (event: L.LeafletMouseEvent) => {
        this.mapClicked.emit({ lat: event.latlng.lat, lng: event.latlng.lng });
      });
      this.render(this.points(), this.cityCenter());
      window.setTimeout(() => this.map?.invalidateSize(), 0);
    });

    effect(() => {
      const points = this.points();
      const cityCenter = this.cityCenter();
      if (this.map) this.render(points, cityCenter);
    });
  }

  private render(points: PoiMapPoint[], cityCenter: { lat: number; lng: number } | null): void {
    if (!this.map) return;
    this.map.stop();
    this.map.invalidateSize();
    this.markers.clearLayers();
    if (!points.length) {
      this.map.setView(cityCenter ? [cityCenter.lat, cityCenter.lng] : [40.4168, -3.7038], cityCenter ? 12 : 5, { animate: false });
      return;
    }

    const bounds = L.latLngBounds([]);
    for (const point of points) {
      const position = L.latLng(point.lat, point.lng);
      bounds.extend(position);
      L.circleMarker(position, {
        radius: 8,
        color: '#fffaf2',
        weight: 4,
        fillColor: '#b86a4b',
        fillOpacity: 1,
      })
        .bindTooltip(point.name, { direction: 'top', offset: [0, -7] })
        .on('click', () => this.poiSelected.emit(point.id))
        .addTo(this.markers);
    }
    if (points.length === 1) {
      this.map.setView([points[0].lat, points[0].lng], 14, { animate: false });
      return;
    }
    const framingPoints = cityCenter ? this.closestToCenter(points, cityCenter, 0.9) : points;
    const framingBounds = L.latLngBounds(
      framingPoints.map((point) => L.latLng(point.lat, point.lng)),
    );
    this.map.fitBounds(framingBounds.isValid() ? framingBounds : bounds, {
      padding: [42, 42],
      maxZoom: 14,
      animate: false,
    });
  }

  private closestToCenter(
    points: PoiMapPoint[],
    center: { lat: number; lng: number },
    ratio: number,
  ): PoiMapPoint[] {
    const count = Math.max(2, Math.ceil(points.length * ratio));
    const lngScale = Math.cos(center.lat * Math.PI / 180);
    return [...points]
      .sort((left, right) => {
        const leftDistance = (left.lat - center.lat) ** 2
          + ((left.lng - center.lng) * lngScale) ** 2;
        const rightDistance = (right.lat - center.lat) ** 2
          + ((right.lng - center.lng) * lngScale) ** 2;
        return leftDistance - rightDistance;
      })
      .slice(0, count);
  }
}
