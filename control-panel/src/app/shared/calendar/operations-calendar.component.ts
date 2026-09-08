import { Component, computed, input } from '@angular/core';
import { FullCalendarModule } from '@fullcalendar/angular';
import { CalendarOptions } from '@fullcalendar/core';
import dayGridPlugin from '@fullcalendar/daygrid';
import interactionPlugin from '@fullcalendar/interaction';

import { CalendarActivity } from '../../core/admin-overview.model';

@Component({
  selector: 'locus-operations-calendar',
  standalone: true,
  imports: [FullCalendarModule],
  template: `
    <section class="calendar-panel">
      <header><div><p>CONVERSACIONES</p><h1>La agenda del sistema.</h1></div><span>{{ activities().length }} sesiones</span></header>
      @if (activities().length) {
        <full-calendar [options]="options()"></full-calendar>
      } @else {
        <div class="empty"><b>Aún no hay sesiones de voz</b><p>Las pruebas y conversaciones aparecerán aquí con su estado y proveedor.</p></div>
      }
    </section>
  `,
  styles: [`
    .calendar-panel { margin-top: 38px; padding: 30px; background: rgba(255,250,242,.78); border: 1px solid var(--locus-line); }
    header { display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 28px; }
    header p { color: var(--locus-blue); font-size: 9px; letter-spacing: .24em; font-weight: 800; margin: 0 0 10px; }
    h1 { font: 600 clamp(36px,4vw,58px)/1 "Fraunces",serif; margin: 0; }
    header span { border: 1px solid var(--locus-line); border-radius: 99px; padding: 9px 13px; font-size: 10px; font-weight: 800; }
    .empty { min-height: 360px; display: grid; place-content: center; text-align: center; background: linear-gradient(135deg,rgba(232,221,199,.28),transparent); border: 1px dashed var(--locus-line); }
    .empty b { font: 600 23px "Fraunces",serif; }.empty p { color: var(--locus-muted); font-size: 11px; }
    :host ::ng-deep .fc { --fc-border-color: rgba(47,93,98,.13); --fc-button-bg-color: #2f5d62; --fc-button-border-color: #2f5d62; --fc-event-bg-color: #b86a4b; --fc-event-border-color: #b86a4b; font-family: "Manrope",sans-serif; font-size: 11px; }
    :host ::ng-deep .fc-toolbar-title { font-family: "Fraunces",serif; font-weight: 600; }
  `],
})
export class OperationsCalendarComponent {
  readonly activities = input.required<CalendarActivity[]>();
  readonly options = computed<CalendarOptions>(() => ({
    plugins: [dayGridPlugin, interactionPlugin],
    initialView: 'dayGridMonth',
    firstDay: 1,
    height: 610,
    events: this.activities().map((event) => ({ id: event.id, title: event.title, start: event.start, classNames: [`status-${event.kind}`] })),
    headerToolbar: { left: 'prev,next today', center: 'title', right: '' },
    buttonText: { today: 'Hoy' },
  }));
}
