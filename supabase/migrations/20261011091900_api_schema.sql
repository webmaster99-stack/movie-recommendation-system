-- Tables behind the API. The Python side (api/recsys_api/db.py) describes the same tables;
-- a test fails if a table or column exists on one side only.
--
-- Access model: the API connects as the database owner and does all the writing. Row-level
-- security is the second layer: it is on for every table, so the public Supabase key that
-- ships in the frontend can read nothing except a signed-in user's own profile and ratings.
--
-- Movies are stored by their MovieLens id (movie_id). The model's own index changes
-- whenever the catalog is rebuilt, so it never reaches the database.

-- One row per account, created by the trigger below when a user signs up.
create table public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  display_name text,
  created_at timestamptz not null default now()
);

create function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, new.raw_user_meta_data ->> 'display_name');
  return new;
end;
$$;

create trigger on_auth_user_created
after insert on auth.users
for each row execute function public.handle_new_user();

-- Only a trigger may call it; without this it would be exposed as an RPC endpoint.
revoke execute on function public.handle_new_user() from public, anon, authenticated;

create table public.ratings (
  user_id uuid not null references auth.users (id) on delete cascade,
  movie_id integer not null,
  rating numeric(2, 1) not null check (rating between 0.5 and 5 and rating * 2 = trunc(rating * 2)),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (user_id, movie_id)
);

-- A user's ratings in time order: the history handed to the model.
create index ratings_user_id_updated_at_idx on public.ratings (user_id, updated_at);

-- Cache of TMDB data. TMDB's terms forbid keeping it for more than 6 months: the API
-- refreshes a row after 150 days and deletes anything older than 180.
create table public.movie_metadata (
  movie_id integer primary key,
  tmdb_id integer not null,
  found boolean not null,
  title text,
  overview text,
  poster_path text,
  backdrop_path text,
  release_date varchar(10),
  runtime integer,
  vote_average double precision,
  fetched_at timestamptz not null
);

create index movie_metadata_fetched_at_idx on public.movie_metadata (fetched_at);

-- One row per answered /recommendations call.
create table public.rec_requests (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  model text not null,
  model_version text not null,
  strategy text not null check (strategy in ('personalised', 'popular')),
  n_liked integer not null,
  movie_ids integer[] not null,
  model_ms double precision not null,
  latency_ms double precision not null,
  created_at timestamptz not null default now()
);

create index rec_requests_user_id_idx on public.rec_requests (user_id);
create index rec_requests_created_at_idx on public.rec_requests (created_at);

-- What users did with recommendations: clicks, and ratings given from a recommended list.
create table public.events (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users (id) on delete cascade,
  request_id uuid references public.rec_requests (id) on delete set null,
  event_type text not null check (event_type in ('click', 'rating')),
  movie_id integer not null,
  position integer check (position >= 0),
  rating numeric(2, 1),
  created_at timestamptz not null default now()
);

create index events_user_id_idx on public.events (user_id);
create index events_request_id_idx on public.events (request_id);
create index events_created_at_idx on public.events (created_at);

alter table public.profiles enable row level security;
alter table public.ratings enable row level security;
alter table public.movie_metadata enable row level security;
alter table public.rec_requests enable row level security;
alter table public.events enable row level security;

-- `(select auth.uid())` instead of `auth.uid()`: evaluated once per query, not once per row.
create policy "Users read their own profile" on public.profiles
  for select to authenticated using ((select auth.uid()) = id);

create policy "Users update their own profile" on public.profiles
  for update to authenticated
  using ((select auth.uid()) = id) with check ((select auth.uid()) = id);

create policy "Users read their own ratings" on public.ratings
  for select to authenticated using ((select auth.uid()) = user_id);

-- No policies on movie_metadata, rec_requests and events: only the API touches them.
