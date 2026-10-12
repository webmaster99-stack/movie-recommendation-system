"use client";

import { SearchIcon, UserRoundIcon } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Suspense } from "react";
import { toast } from "sonner";
import { useAuth } from "@/components/auth-provider";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { APP_NAME } from "@/lib/config";
import { supabase } from "@/lib/supabase";
import { cn } from "@/lib/utils";

const SIGNED_IN_LINKS = [
  { href: "/for-you", label: "For you" },
  { href: "/ratings", label: "My ratings" },
];
const ALWAYS_LINKS = [{ href: "/about", label: "How it works" }];

type NavLink = { href: string; label: string };

function NavLinks({ links, current }: { links: NavLink[]; current?: string }) {
  return links.map(({ href, label }) => (
    <Link
      key={href}
      href={href}
      aria-current={current === href ? "page" : undefined}
      className="rounded-sm py-1 text-muted-foreground hover:text-foreground aria-[current=page]:text-foreground aria-[current=page]:underline aria-[current=page]:decoration-primary aria-[current=page]:decoration-2 aria-[current=page]:underline-offset-8"
    >
      {label}
    </Link>
  ));
}

// The path of a movie page isn't known when the page's shell is prerendered, so reading it
// has to sit behind Suspense; until it resolves the links render without a current page.
function CurrentNavLinks({ links }: { links: NavLink[] }) {
  return <NavLinks links={links} current={usePathname()} />;
}

function AccountMenu({ email }: { email: string | undefined }) {
  const router = useRouter();

  async function logOut() {
    const { error } = await supabase().auth.signOut();
    if (error) {
      toast.error(`Couldn't log out. ${error.message}`);
      return;
    }
    router.push("/");
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon-lg" aria-label="Account">
          <UserRoundIcon />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-48">
        <DropdownMenuLabel className="truncate">{email ?? "Signed in"}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem asChild>
          <Link href="/onboarding">Rate more movies</Link>
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={logOut}>Log out</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function SiteHeader() {
  const { status, user } = useAuth();
  const links = [...(status === "signedIn" ? SIGNED_IN_LINKS : []), ...ALWAYS_LINKS];

  return (
    <header className="border-b">
      <div className="container-page flex flex-wrap items-center gap-x-7 gap-y-1 py-3">
        <Link
          href={status === "signedIn" ? "/for-you" : "/"}
          className="font-heading text-2xl font-extrabold tracking-wide uppercase"
        >
          {APP_NAME}
        </Link>
        <nav
          aria-label="Main"
          className="order-last flex w-full gap-6 text-sm sm:order-none sm:w-auto"
        >
          <Suspense fallback={<NavLinks links={links} />}>
            <CurrentNavLinks links={links} />
          </Suspense>
        </nav>
        <div className={cn("ml-auto flex items-center gap-1", status === "loading" && "invisible")}>
          <Button variant="ghost" size="icon-lg" asChild>
            <Link href="/search" aria-label="Search movies">
              <SearchIcon />
            </Link>
          </Button>
          {status === "signedIn" ? (
            <AccountMenu email={user?.email} />
          ) : (
            <>
              <Button variant="ghost" size="lg" asChild>
                <Link href="/login">Log in</Link>
              </Button>
              <Button size="lg" asChild>
                <Link href="/signup">Sign up</Link>
              </Button>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
