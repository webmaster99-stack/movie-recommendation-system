import { API_URL } from "@/lib/config";
import { supabase } from "@/lib/supabase";
import type { components } from "./schema";

// The types come from the API's own schema (`npm run gen:api`), so a change to an endpoint
// that this code doesn't follow is a type error.
type Schemas = components["schemas"];
export type Health = Schemas["Health"];
export type Movie = Schemas["MovieOut"];
export type MovieDetail = Schemas["MovieDetail"];
export type OnboardingMovies = Schemas["OnboardingMovies"];
export type RatedMovie = Schemas["RatedMovie"];
export type RatingsPage = Schemas["RatingsPage"];
export type Recommendations = Schemas["Recommendations"];

export class ApiError extends Error {
  /** The HTTP status, or 0 when no response arrived at all. */
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * True for the failures seen while the free host starts the API after it has been idle:
 * no response, or a gateway error from the host's proxy. These are worth retrying.
 */
export function isWakingUp(error: unknown): boolean {
  return error instanceof ApiError && [0, 502, 503, 504].includes(error.status);
}

type RequestOptions = {
  method?: "GET" | "POST" | "DELETE";
  query?: Record<string, string | number>;
  body?: unknown;
  /** Send the signed-in user's access token. */
  auth?: boolean;
  /** Let the request finish after the page is left (for click logging). */
  keepalive?: boolean;
};

async function accessToken(): Promise<string> {
  // getSession() refreshes an expired token before returning it.
  const { data } = await supabase().auth.getSession();
  if (!data.session) throw new ApiError(401, "Log in to continue.");
  return data.session.access_token;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const { detail } = await response.json();
    if (typeof detail === "string") return detail;
  } catch {
    // Not JSON: a gateway's error page, for instance.
  }
  return `The server answered ${response.status}.`;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const url = new URL(API_URL + path);
  for (const [name, value] of Object.entries(options.query ?? {})) {
    url.searchParams.set(name, String(value));
  }
  const headers: Record<string, string> = {};
  if (options.auth) headers.Authorization = `Bearer ${await accessToken()}`;
  if (options.body !== undefined) headers["Content-Type"] = "application/json";

  let response: Response;
  try {
    response = await fetch(url, {
      method: options.method ?? "GET",
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      keepalive: options.keepalive,
    });
  } catch {
    throw new ApiError(0, "The server can't be reached.");
  }
  if (!response.ok) throw new ApiError(response.status, await errorMessage(response));
  // 201 and 204 carry no body.
  return (response.status === 200 ? await response.json() : undefined) as T;
}

export const getHealth = () => request<Health>("/health");

export const searchMovies = (q: string, limit = 30) =>
  request<Movie[]>("/movies/search", { query: { q, limit } });

export const getMovie = (movieId: number) => request<MovieDetail>(`/movies/${movieId}`);

export const getSimilarMovies = (movieId: number, k = 12) =>
  request<Movie[]>(`/movies/${movieId}/similar`, { query: { k } });

export const getOnboardingMovies = (limit: number, offset: number) =>
  request<OnboardingMovies>("/onboarding/movies", { query: { limit, offset } });

export const getRecommendations = (k = 20) =>
  request<Recommendations>("/recommendations", { query: { k }, auth: true });

const RATINGS_PAGE_SIZE = 200; // the most the API returns at once

/** Every rating of the signed-in user, newest first. */
export async function getAllRatings(): Promise<RatedMovie[]> {
  const items: RatedMovie[] = [];
  for (let offset = 0; ; offset += RATINGS_PAGE_SIZE) {
    const page = await request<RatingsPage>("/ratings", {
      query: { limit: RATINGS_PAGE_SIZE, offset },
      auth: true,
    });
    items.push(...page.items);
    if (offset + RATINGS_PAGE_SIZE >= page.total) return items;
  }
}

export const rateMovie = (movieId: number, rating: number, requestId?: string) =>
  request<Schemas["RatingOut"]>("/ratings", {
    method: "POST",
    body: { movie_id: movieId, rating, request_id: requestId ?? null },
    auth: true,
  });

export const deleteRating = (movieId: number) =>
  request<void>(`/ratings/${movieId}`, { method: "DELETE", auth: true });

/** Record that a movie was opened, with the recommendation list and place it came from. */
export const logClick = (movieId: number, requestId?: string, position?: number) =>
  request<void>("/events", {
    method: "POST",
    body: {
      event_type: "click",
      movie_id: movieId,
      request_id: requestId ?? null,
      position: position ?? null,
    },
    auth: true,
    keepalive: true,
  });
