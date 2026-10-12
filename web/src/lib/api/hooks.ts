"use client";

import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";
import { toast } from "sonner";
import { useAuth } from "@/components/auth-provider";
import {
  deleteRating,
  getAllRatings,
  getHealth,
  getMovie,
  getOnboardingMovies,
  getRecommendations,
  getSimilarMovies,
  rateMovie,
  searchMovies,
  type Movie,
  type RatedMovie,
} from "./client";
import { applyRating } from "./ratings";

const ONBOARDING_PAGE_SIZE = 40;

export const keys = {
  health: ["health"] as const,
  movie: (movieId: number) => ["movie", movieId] as const,
  similar: (movieId: number, k: number) => ["similar", movieId, k] as const,
  search: (q: string) => ["search", q] as const,
  onboarding: ["onboarding"] as const,
  minRatings: ["min-ratings"] as const,
  // Per user, so one account never sees what another left in the cache.
  ratings: (userId: string | undefined) => ["ratings", userId] as const,
  recommendations: (userId: string | undefined, k: number) =>
    ["recommendations", userId, k] as const,
};

export function useHealth() {
  return useQuery({ queryKey: keys.health, queryFn: getHealth, staleTime: Infinity });
}

export function useMovie(movieId: number) {
  return useQuery({ queryKey: keys.movie(movieId), queryFn: () => getMovie(movieId) });
}

export function useSimilarMovies(movieId: number | undefined, k = 12) {
  return useQuery({
    queryKey: keys.similar(movieId ?? 0, k),
    queryFn: () => getSimilarMovies(movieId as number, k),
    enabled: movieId !== undefined,
  });
}

export function useSearch(q: string) {
  return useQuery({
    queryKey: keys.search(q),
    queryFn: () => searchMovies(q),
    enabled: q.length > 0,
    // Keep showing the last results while the next keystroke's are on their way.
    placeholderData: (previous) => previous,
  });
}

export function useOnboardingMovies() {
  return useInfiniteQuery({
    queryKey: keys.onboarding,
    queryFn: ({ pageParam }) => getOnboardingMovies(ONBOARDING_PAGE_SIZE, pageParam),
    initialPageParam: 0,
    getNextPageParam: (last, pages) =>
      last.movies.length < ONBOARDING_PAGE_SIZE ? undefined : pages.length * ONBOARDING_PAGE_SIZE,
    staleTime: Infinity,
  });
}

/** How many ratings the app asks for before it shows recommendations. */
export function useMinRatings() {
  return useQuery({
    queryKey: keys.minRatings,
    queryFn: async () => (await getOnboardingMovies(1, 0)).min_ratings,
    staleTime: Infinity,
  });
}

/** All of the signed-in user's ratings, newest first. Idle while signed out. */
export function useRatings() {
  const { user } = useAuth();
  return useQuery({
    queryKey: keys.ratings(user?.id),
    queryFn: getAllRatings,
    enabled: user !== undefined,
  });
}

/** The user's rating of one movie; undefined when unrated or signed out. */
export function useMyRating(movieId: number): number | undefined {
  const { data } = useRatings();
  return data?.find((item) => item.movie.movie_id === movieId)?.rating;
}

export function useRecommendations(k = 20, enabled = true) {
  const { user } = useAuth();
  return useQuery({
    queryKey: keys.recommendations(user?.id, k),
    queryFn: () => getRecommendations(k),
    enabled: enabled && user !== undefined,
    // A fresh list on every visit, since any rating changes it, but never while the page
    // is being read: a list that reorders itself under the cursor is worse than a stale one.
    staleTime: 0,
    refetchOnMount: "always",
  });
}

type RatingChange = {
  movie: Movie;
  /** `null` removes the rating. */
  rating: number | null;
  /** The recommendation list the movie was opened from, to credit it with the rating. */
  requestId?: string;
};

const RATING_MUTATION = ["rating"] as const;

function ratingsSettled(queryClient: QueryClient, userId: string | undefined) {
  // With several changes in flight (ten quick ratings while onboarding), only the last
  // one refetches; an earlier refetch would briefly undo the later optimistic changes.
  if (queryClient.isMutating({ mutationKey: RATING_MUTATION }) > 1) return;
  void queryClient.invalidateQueries({ queryKey: keys.ratings(userId) });
  void queryClient.invalidateQueries({ queryKey: ["recommendations"], refetchType: "none" });
}

/** Rate a movie, change its rating or remove it. The stars update before the server answers. */
export function useRateMovie() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const ratingsKey = keys.ratings(user?.id);

  return useMutation({
    mutationKey: RATING_MUTATION,
    mutationFn: async ({ movie, rating, requestId }: RatingChange) => {
      if (rating === null) await deleteRating(movie.movie_id);
      else await rateMovie(movie.movie_id, rating, requestId);
    },
    onMutate: async ({ movie, rating }) => {
      await queryClient.cancelQueries({ queryKey: ratingsKey });
      const previous = queryClient.getQueryData<RatedMovie[]>(ratingsKey);
      queryClient.setQueryData<RatedMovie[]>(ratingsKey, (current) =>
        applyRating(current ?? [], movie, rating),
      );
      return { previous };
    },
    onError: (error, { movie, rating }, context) => {
      queryClient.setQueryData(ratingsKey, context?.previous);
      const what = rating === null ? "remove your rating of" : "save your rating of";
      toast.error(`Couldn't ${what} ${movie.title}. ${error.message}`);
    },
    onSettled: () => ratingsSettled(queryClient, user?.id),
  });
}
