import { Component, computed, input } from '@angular/core';
import { GoogleMapsModule } from '@angular/google-maps';

import { PoiMapPoint } from '../../core/admin-overview.model';

@Component({
  selector: 'locus-poi-map',
  standalone: true,
  imports: [GoogleMapsModule],
  template: `
    <section class="map-panel">
      <header><div><p>CATÁLOGO GEOGRÁFICO</p><h1>Ciudades y lugares.</h1></div><span>{{ points().length }} POIs visibles</span></header>
      @if (mapsAvailable && points().length) {
        <google-map height="620px" width="100%" [center]="center()" [zoom]="11" [options]="mapOptions">
          @for (point of points(); track point.id) {
            <map-advanced-marker [position]="{lat: point.lat, lng: point.lng}" [title]="point.name" />
          }
        </google-map>
      } @else {
        <div class="atlas" aria-label="Vista previa del catálogo de puntos de interés">
          <div class="grid"></div><div class="ring one"></div><div class="ring two"></div>
          @for (point of previewPoints(); track point.id; let index = $index) {
            <button [style.left.%]="point.x" [style.top.%]="point.y" [title]="point.name"><i></i><span>{{ point.name }}</span></button>
          }
          <div class="map-empty"><b>{{ points().length ? 'Vista cartográfica preparada' : 'Esperando el catálogo V1' }}</b><p>{{ points().length ? 'Añade la clave web restringida de Google Maps para activar el mapa.' : 'Los POIs aparecerán después de la importación.' }}</p></div>
        </div>
      }
    </section>
  `,
  styles: [`
    .map-panel { margin-top: 38px; padding: 30px; background: rgba(255,250,242,.78); border: 1px solid var(--locus-line); }
    header { display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 28px; }
    header p { color: var(--locus-blue); font-size: 9px; letter-spacing: .24em; font-weight: 800; margin: 0 0 10px; }
    h1 { font: 600 clamp(36px,4vw,58px)/1 "Fraunces",serif; margin: 0; }
    header span { border: 1px solid var(--locus-line); border-radius: 99px; padding: 9px 13px; font-size: 10px; font-weight: 800; }
    .atlas { height: 620px; position: relative; overflow: hidden; background: #e6e4d7; border: 1px solid rgba(47,93,98,.16); }
    .grid { position:absolute;inset:0;background-image:linear-gradient(rgba(47,93,98,.09) 1px,transparent 1px),linear-gradient(90deg,rgba(47,93,98,.09) 1px,transparent 1px);background-size:62px 62px;transform:rotate(-7deg) scale(1.2); }
    .ring { position:absolute;border:1px solid rgba(184,106,75,.25);border-radius:50%; }.ring.one{width:420px;height:420px;right:-100px;top:-120px}.ring.two{width:270px;height:270px;left:-70px;bottom:-80px}
    .atlas button { position:absolute;transform:translate(-50%,-100%);border:0;background:transparent;color:var(--locus-ink);font-size:9px;cursor:pointer;z-index:2; }
    .atlas button i { display:block;width:22px;height:22px;margin:auto;background:var(--locus-terracotta);border:5px solid var(--locus-paper);border-radius:50% 50% 50% 0;transform:rotate(-45deg);box-shadow:0 4px 12px rgba(31,42,46,.2); }
    .atlas button span { display:none;white-space:nowrap;background:var(--locus-paper);padding:5px 8px;border-radius:8px;box-shadow:0 5px 20px rgba(31,42,46,.14); }.atlas button:hover span{display:block}
    .map-empty { position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);padding:28px 34px;background:rgba(255,250,242,.92);text-align:center;z-index:3;box-shadow:var(--locus-shadow); }
    .map-empty b { font:600 22px "Fraunces",serif; }.map-empty p{max-width:330px;color:var(--locus-muted);font-size:11px;line-height:1.6;margin:8px 0 0}
  `],
})
export class PoiMapComponent {
  readonly points = input.required<PoiMapPoint[]>();
  readonly mapsAvailable = typeof window !== 'undefined' && 'google' in window;
  readonly mapOptions = { mapId: 'DEMO_MAP_ID', disableDefaultUI: false };
  readonly center = computed(() => this.points().length ? { lat: this.points()[0].lat, lng: this.points()[0].lng } : { lat: 40.4168, lng: -3.7038 });
  readonly previewPoints = computed(() => this.points().slice(0, 24).map((point, index) => ({ ...point, x: 10 + ((index * 37) % 80), y: 15 + ((index * 53) % 72) })));
}
