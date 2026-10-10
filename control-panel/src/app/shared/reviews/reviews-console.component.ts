import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, computed, inject, signal } from '@angular/core';

export interface AdminReview {
  id: number;
  call_id: string;
  created_at: string;
  user_email: string;
  user_name: string;
  poi_name: string;
  place_vote: boolean | null;
  place_comment: string | null;
  place_comment_status: 'published' | 'pending' | 'rejected' | null;
  guide_stars: number | null;
  guide_tags: string[];
  guide_comment: string | null;
}

const TAG_LABELS: Record<string, string> = {
  learned: 'Contó cosas que no sabía',
  too_long: 'Se enrolló demasiado',
  wrong_fact: 'Se equivocó en algún dato',
  cut_off: 'Se cortaba',
  misheard: 'No me entendía',
};

const STATUS_LABELS: Record<string, string> = {
  published: 'Publicado',
  pending: 'Pendiente',
  rejected: 'Rechazado',
};

/**
 * Lo que la gente opina al colgar. La parte del guía solo se ve aquí: es la
 * forma de saber cómo se comporta en llamadas reales sin escucharlas.
 */
@Component({
  selector: 'locus-reviews-console',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './reviews-console.component.html',
  styleUrl: './reviews-console.component.scss',
})
export class ReviewsConsoleComponent {
  private readonly http = inject(HttpClient);

  readonly reviews = signal<AdminReview[]>([]);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);

  readonly averageStars = computed(() => {
    const notas = this.reviews().map((r) => r.guide_stars).filter((n): n is number => n !== null);
    return notas.length ? notas.reduce((a, b) => a + b, 0) / notas.length : null;
  });
  readonly likeRatio = computed(() => {
    const votos = this.reviews().map((r) => r.place_vote).filter((v): v is boolean => v !== null);
    return votos.length ? votos.filter(Boolean).length / votos.length : null;
  });
  readonly unpublished = computed(
    () => this.reviews().filter((r) => r.place_comment && r.place_comment_status !== 'published').length,
  );
  readonly tagCounts = computed(() => {
    const cuenta = new Map<string, number>();
    for (const review of this.reviews()) {
      for (const tag of review.guide_tags) {
        cuenta.set(tag, (cuenta.get(tag) ?? 0) + 1);
      }
    }
    return Object.keys(TAG_LABELS).map((tag) => ({ tag, label: TAG_LABELS[tag], count: cuenta.get(tag) ?? 0 }));
  });
  readonly maxTag = computed(() => Math.max(1, ...this.tagCounts().map((t) => t.count)));

  constructor() {
    this.load();
  }

  load(): void {
    this.loading.set(true);
    this.error.set(null);
    this.http.get<AdminReview[]>('/admin/v2/reviews').subscribe({
      next: (reviews) => {
        this.reviews.set(reviews);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido leer las valoraciones.');
        this.loading.set(false);
      },
    });
  }

  tagLabel(tag: string): string {
    return TAG_LABELS[tag] ?? tag;
  }

  statusLabel(status: string | null): string {
    return status ? STATUS_LABELS[status] ?? status : '';
  }

  stars(n: number | null): string {
    return n ? '★'.repeat(n) + '☆'.repeat(5 - n) : '—';
  }
}
