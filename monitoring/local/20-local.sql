-- For the local monitoring stack only (monitoring/docker-compose.yml); never run this
-- against Supabase.

-- The password Grafana connects with. In Supabase you choose one yourself.
alter role monitoring_reader with password 'monitoring';

-- The load test (api/loadtest/locustfile.py) makes up its users, so here no account has to
-- exist in auth.users before it can rate movies.
alter table public.ratings drop constraint ratings_user_id_fkey;
alter table public.rec_requests drop constraint rec_requests_user_id_fkey;
alter table public.events drop constraint events_user_id_fkey;
