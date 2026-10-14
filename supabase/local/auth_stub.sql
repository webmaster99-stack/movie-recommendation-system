-- The parts of Supabase that the migrations refer to, for a plain Postgres: the tests
-- (TEST_POSTGRES_URL) and the local monitoring stack (monitoring/docker-compose.yml) run
-- this first and the files in ../migrations after it. Never run it against Supabase.

create schema auth;

create table auth.users (
  id uuid primary key,
  raw_user_meta_data jsonb
);

-- In Supabase: the id of the signed-in user, taken from the request's token.
create function auth.uid() returns uuid
language sql stable
as $$ select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;

-- The roles the frontend's key and a signed-in user connect as.
-- Roles belong to the whole server, not to one database, so they may exist already.
do $$
begin
  if not exists (select from pg_roles where rolname = 'anon') then
    create role anon nologin;
  end if;
  if not exists (select from pg_roles where rolname = 'authenticated') then
    create role authenticated nologin;
  end if;
end
$$;
