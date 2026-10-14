-- Monitoring: a log of every request, two more facts per recommendation list, and a
-- read-only window onto all of it for the dashboards (Grafana) and the weekly drift report.
--
-- Access model: Grafana Cloud and the GitHub workflow connect as `monitoring_reader`, a role
-- that can read the views in the `monitoring` schema and nothing else. The views carry no
-- user ids, so a leaked monitoring password exposes how the app is used, not who uses it.

-- One row per answered request, queued by the API and written in batches by a background
-- thread. The API deletes rows older than 90 days (REQUEST_LOG_DAYS).
create table public.request_log (
  id bigint generated always as identity primary key,
  method text not null,
  route text not null,
  status integer not null,
  latency_ms double precision not null,
  created_at timestamptz not null default now()
);

create index request_log_created_at_idx on public.request_log (created_at);

alter table public.request_log enable row level security;
-- No policies: only the API touches it.

-- Popularity bias and coverage of what is served. `popularity_percentile` is the mean place
-- of the shown movies in the model's popularity order, from 0 (the most popular movie) to 1
-- (the least). Both are null on lists served before this migration.
alter table public.rec_requests
  add column popularity_percentile double precision,
  add column catalog_size integer;

-- Not `public`: Supabase's REST API exposes only the schemas listed in its settings, so
-- nothing here is reachable with the frontend's key.
create schema monitoring;

-- A view runs with its owner's rights, which is what lets `monitoring_reader` read through
-- it without any access to the tables underneath, and what makes the column list of each
-- view the whole of what that role can ever see.

create view monitoring.requests as
select created_at, method, route, status, latency_ms
from public.request_log;

-- A user appears as `user_key`, an MD5 of the account id: enough to count people and to
-- group one person's ratings, useless for finding the account (the ids are random UUIDs
-- that `monitoring_reader` cannot read anywhere).
create view monitoring.ratings as
select md5(user_id::text) as user_key, movie_id, rating, created_at, updated_at
from public.ratings;

-- One row per recommendation list, with what became of it.
create view monitoring.recommendations as
select
  r.created_at,
  md5(r.user_id::text) as user_key,
  r.model,
  r.model_version,
  r.strategy,
  r.n_liked,
  cardinality(r.movie_ids) as n_items,
  r.model_ms,
  r.latency_ms,
  r.popularity_percentile,
  r.catalog_size,
  -- Clicks and ratings that the app credited to this list.
  coalesce(e.n_clicks, 0) as n_clicks,
  coalesce(e.n_ratings, 0) as n_ratings,
  e.first_click_position,
  -- The highest rating the user later gave to any movie on the list, however they got to
  -- it; null when they rated none of them. "Hit rate" is the share of lists where it is set.
  (
    select max(x.rating)
    from public.ratings x
    where x.user_id = r.user_id
      and x.movie_id = any (r.movie_ids)
      and x.updated_at > r.created_at
  ) as best_rating_after
from public.rec_requests r
left join lateral (
  select
    count(*) filter (where e.event_type = 'click') as n_clicks,
    count(*) filter (where e.event_type = 'rating') as n_ratings,
    min(e.position) filter (where e.event_type = 'click') as first_click_position
  from public.events e
  where e.request_id = r.id
) e on true;

-- One row per movie shown: what coverage and the genre mix of the served lists are
-- computed from. `position` is 0-based, as in `events`.
create view monitoring.served_movies as
select
  r.created_at,
  r.model_version,
  r.strategy,
  m.movie_id,
  (m.ordinality - 1)::integer as position
from public.rec_requests r
cross join lateral unnest(r.movie_ids) with ordinality as m (movie_id, ordinality);

create view monitoring.events as
select
  created_at,
  event_type,
  movie_id,
  position,
  rating,
  request_id is not null as from_recommendation
from public.events;

-- Created without a password, so nobody can connect as it until one is set:
--   alter role monitoring_reader with password '...';
do $$
begin
  if not exists (select from pg_roles where rolname = 'monitoring_reader') then
    create role monitoring_reader login;
  end if;
end
$$;

-- A dashboard query that runs away must not hold the small database hostage.
alter role monitoring_reader set statement_timeout = '30s';

grant usage on schema monitoring to monitoring_reader;
grant select on all tables in schema monitoring to monitoring_reader;
alter default privileges in schema monitoring grant select on tables to monitoring_reader;
