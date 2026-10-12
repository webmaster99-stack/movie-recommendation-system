import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  getAllRatings,
  getHealth,
  getRecommendations,
  isWakingUp,
  logClick,
  rateMovie,
  searchMovies,
  type RatedMovie,
} from "./client";

const session = vi.hoisted(() => ({
  current: { access_token: "token-1" } as { access_token: string } | null,
}));
vi.mock("@/lib/supabase", () => ({
  supabase: () => ({
    auth: { getSession: async () => ({ data: { session: session.current } }) },
  }),
}));

const fetchMock = vi.fn<typeof fetch>();

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

/** The URL and options of the n-th call to fetch. */
function call(n = 0): { url: URL; init: RequestInit } {
  const [url, init] = fetchMock.mock.calls[n];
  return { url: url as URL, init: init as RequestInit };
}

beforeEach(() => {
  session.current = { access_token: "token-1" };
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("requests", () => {
  it("sends public requests without a token", async () => {
    fetchMock.mockResolvedValue(json([]));
    await searchMovies("the matrix", 5);

    const { url, init } = call();
    expect(url.href).toBe("http://127.0.0.1:8000/movies/search?q=the+matrix&limit=5");
    expect(init.headers).toEqual({});
  });

  it("sends the access token where the API asks for a user", async () => {
    fetchMock.mockResolvedValue(json({ items: [] }));
    await getRecommendations(20);

    expect(call().init.headers).toEqual({ Authorization: "Bearer token-1" });
  });

  it("does not call the API for a user who is signed out", async () => {
    session.current = null;

    await expect(getRecommendations()).rejects.toMatchObject({ status: 401 });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("posts a rating with the list it came from", async () => {
    fetchMock.mockResolvedValue(json({ movie_id: 10, rating: 4.5, updated_at: "x" }));
    await rateMovie(10, 4.5, "list-1");

    const { init } = call();
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      movie_id: 10,
      rating: 4.5,
      request_id: "list-1",
    });
  });

  it("logs a click so that it survives leaving the page, and accepts the empty answer", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 201 }));

    await expect(logClick(10, "list-1", 3)).resolves.toBeUndefined();
    expect(call().init.keepalive).toBe(true);
    expect(JSON.parse(call().init.body as string)).toEqual({
      event_type: "click",
      movie_id: 10,
      request_id: "list-1",
      position: 3,
    });
  });
});

describe("errors", () => {
  it("reports the API's own explanation", async () => {
    fetchMock.mockResolvedValue(json({ detail: "no movie with id 99" }, 404));

    const error = await getHealth().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 404, message: "no movie with id 99" });
  });

  it("copes with an error that is not JSON", async () => {
    fetchMock.mockResolvedValue(new Response("<html>Bad Gateway</html>", { status: 502 }));

    await expect(getHealth()).rejects.toMatchObject({ status: 502 });
  });

  it("turns a failed connection into status 0", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(getHealth()).rejects.toMatchObject({ status: 0 });
  });

  it("retries only what a sleeping server causes", () => {
    for (const status of [0, 502, 503, 504]) {
      expect(isWakingUp(new ApiError(status, ""))).toBe(true);
    }
    for (const status of [401, 404, 422, 500]) {
      expect(isWakingUp(new ApiError(status, ""))).toBe(false);
    }
    expect(isWakingUp(new Error("anything else"))).toBe(false);
  });
});

describe("getAllRatings", () => {
  const rated = (movieId: number) => ({ movie: { movie_id: movieId }, rating: 4 }) as RatedMovie;

  it("reads every page", async () => {
    const first = Array.from({ length: 200 }, (_, i) => rated(i));
    fetchMock
      .mockResolvedValueOnce(json({ total: 201, n_liked: 201, items: first }))
      .mockResolvedValueOnce(json({ total: 201, n_liked: 201, items: [rated(200)] }));

    const all = await getAllRatings();

    expect(all).toHaveLength(201);
    expect(call(0).url.search).toBe("?limit=200&offset=0");
    expect(call(1).url.search).toBe("?limit=200&offset=200");
  });

  it("stops after one page when that is all there is", async () => {
    // `total` also counts ratings of movies that have left the catalog, which the API hides.
    fetchMock.mockResolvedValue(json({ total: 3, n_liked: 2, items: [rated(1), rated(2)] }));

    expect(await getAllRatings()).toHaveLength(2);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
