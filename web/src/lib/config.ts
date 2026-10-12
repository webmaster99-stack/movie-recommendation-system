// A working title; this is the only place it is written.
export const APP_NAME = "Double Feature";

export const REPO_URL = "https://github.com/webmaster99-stack/movie-recommendation-system";

// 127.0.0.1 rather than localhost: on Windows each new connection to `localhost` can wait
// for IPv6 to fail first.
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(
  /\/+$/,
  "",
);
